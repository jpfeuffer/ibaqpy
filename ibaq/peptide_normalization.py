#!/usr/bin/env python
import numpy as np
import pandas as pd
import os
import re
import duckdb
from ibaq.normalization_methods import normalize_run, normalize
from ibaq.ibaqpy_commons import (
    BIOREPLICATE,
    CHANNEL,
    CONDITION,
    PARQUET_COLUMNS,
    FRACTION,
    FRAGMENT_ION,
    INTENSITY,
    ISOTOPE_LABEL_TYPE,
    NORM_INTENSITY,
    PEPTIDE_CANONICAL,
    PEPTIDE_CHARGE,
    PEPTIDE_SEQUENCE,
    PROTEIN_NAME,
    REFERENCE,
    RUN,
    SAMPLE_ID,
    SEARCH_ENGINE,
    STUDY_ID,
    TMT16plex,
    TMT11plex,
    TMT10plex,
    TMT6plex,
    ITRAQ4plex,
    ITRAQ8plex,
    parquet_map,
    print_help_msg,
)


def get_spectrum_prefix(reference_spectrum: str) -> str:
    """
    Get the reference name from Reference column. The function expected a reference name in the following format eg.
    20150820_Haura-Pilot-TMT1-bRPLC03-2.mzML_controllerType=0 controllerNumber=1 scan=16340. This function can also
    remove suffix of spectrum files.
    :param reference_spectrum:
    :return: reference name
    """
    return re.split(r"\.mzML|\.MZML|\.raw|\.RAW|\.d|\.wiff", reference_spectrum)[0]


def get_study_accession(sample_id: str) -> str:
    """
    Get the project accession from the Sample accession. The function expected a sample accession in the following
    format PROJECT-SAMPLEID
    :param sample_id: Sample Accession
    :return: study accession
    """
    return sample_id.split("-")[0]


def parse_uniprot_accession(uniprot_id: str) -> str:
    """
    Parse the uniprot accession from the uniprot id in the form of
    tr|CONTAMINANT_Q3SX28|CONTAMINANT_TPM2_BOVIN and convert to CONTAMINANT_Q3SX28
    :param uniprot_id: uniprot id
    :return: uniprot accession
    """
    uniprot_list = uniprot_id.split(";")
    result_uniprot_list = []
    for accession in uniprot_list:
        if accession.count("|") == 2:
            accession = accession.split("|")[1]
        result_uniprot_list.append(accession)
    return ";".join(result_uniprot_list)


def get_canonical_peptide(peptide_sequence: str) -> str:
    """
    This function returns a peptide sequence without the modification information
    :param peptide_sequence: peptide sequence with mods
    :return: peptide sequence
    """
    clean_peptide = re.sub("[\(\[].*?[\)\]]", "", peptide_sequence)
    clean_peptide = clean_peptide.replace(".", "").replace("-", "")
    return clean_peptide


def analyse_sdrf(sdrf_path: str) -> tuple:
    """
    This function is aimed to parse SDRF and return four objects:
    1. sdrf_df: A dataframe with channels and references annoted.
    2. label: Label type of the experiment. LFQ, TMT or iTRAQ.
    3. sample_names: A list contains all sample names.
    4. choice: A dictionary caontains key-values between channel
        names and numbers.
    :param sdrf_path: File path of SDRF.
    :param compression: Whether compressed.
    :return:
    """
    sdrf_df = pd.read_csv(sdrf_path, sep="\t")
    sdrf_df.columns = [i.lower() for i in sdrf_df.columns]
    sdrf_df[REFERENCE] = sdrf_df["comment[data file]"].apply(get_spectrum_prefix)

    labels = set(sdrf_df["comment[label]"])
    # Determine label type
    label, choice = get_label(labels)
    if label == "TMT":
        choice_df = (
            pd.DataFrame.from_dict(choice, orient="index", columns=[CHANNEL])
            .reset_index()
            .rename(columns={"index": "comment[label]"})
        )
        sdrf_df = sdrf_df.merge(choice_df, on="comment[label]", how="left")
    elif label == "ITRAQ":
        choice_df = (
            pd.DataFrame.from_dict(choice, orient="index", columns=[CHANNEL])
            .reset_index()
            .rename(columns={"index": "comment[label]"})
        )
        sdrf_df = sdrf_df.merge(choice_df, on="comment[label]", how="left")
    sample_names = sdrf_df["source name"].unique().tolist()
    technical_repetitions = len(sdrf_df["comment[technical replicate]"].unique())

    return technical_repetitions, label, sample_names, choice


def get_label(labels: list) -> (str, dict):
    """Return label type and choice dict according to labels list.

    :param labels: Labels from SDRF.
    :return: Tuple contains label type and choice dict.
    """
    choice = None
    if len(labels) == 1:
        label = "LFQ"
    elif "TMT" in ",".join(labels) or "tmt" in ",".join(labels):
        if (
            len(labels) > 11
            or "TMT134N" in labels
            or "TMT133C" in labels
            or "TMT133N" in labels
            or "TMT132C" in labels
            or "TMT132N" in labels
        ):
            choice = TMT16plex
        elif len(labels) == 11 or "TMT131C" in labels:
            choice = TMT11plex
        elif len(labels) > 6:
            choice = TMT10plex
        else:
            choice = TMT6plex
        label = "TMT"
    elif "ITRAQ" in ",".join(labels) or "itraq" in ",".join(labels):
        if len(labels) > 4:
            choice = ITRAQ8plex
        else:
            choice = ITRAQ4plex
        label = "ITRAQ"
    else:
        exit("Warning: Only support label free, TMT and ITRAQ experiment!")
    return label, choice


def remove_contaminants_entrapments_decoys(
    dataset: pd.DataFrame, protein_field=PROTEIN_NAME
) -> pd.DataFrame:
    """
    This method reads a file with a list of contaminants and high abudant proteins and
    remove them from the dataset.
    :param dataset: Peptide intensity DataFrame
    :param contaminants_file: contaminants file
    :param protein_field: protein field
    :return: dataset with the filtered proteins
    """
    contaminants = []
    contaminants.append("CONTAMINANT")
    contaminants.append("ENTRAPMENT")
    contaminants.append("DECOY")
    cregex = "|".join(contaminants)
    return dataset[~dataset[protein_field].str.contains(cregex)]


def remove_protein_by_ids(
    dataset: pd.DataFrame, protein_file: str, protein_field=PROTEIN_NAME
) -> pd.DataFrame:
    """
    This method reads a file with a list of contaminants and high abudant proteins and
    remove them from the dataset.
    :param dataset: Peptide intensity DataFrame
    :param protein_file: contaminants file
    :param protein_field: protein field
    :return: dataset with the filtered proteins
    """
    contaminants_reader = open(protein_file, "r")
    contaminants = contaminants_reader.read().split("\n")
    contaminants = [cont for cont in contaminants if cont.strip()]
    cregex = "|".join(contaminants)
    return dataset[~dataset[protein_field].str.contains(cregex)]


def parquet_common_process(
    data_df: pd.DataFrame, label: str, choice: dict
) -> pd.DataFrame:
    """Apply common process on data.

    :param data_df: Feature data in dataframe.
    :return: Processed data.
    """
    data_df = data_df.rename(columns=parquet_map)
    data_df[PROTEIN_NAME] = data_df.apply(lambda x: ";".join(x[PROTEIN_NAME]), axis=1)
    if label == "LFQ":
        data_df.drop(CHANNEL, inplace=True, axis=1)
    else:
        data_df[CHANNEL] = data_df[CHANNEL].map(choice)

    return data_df


def data_common_process(data_df: pd.DataFrame, min_aa: int) -> pd.DataFrame:
    # Remove 0 intensity signals from the data
    data_df = data_df[data_df[INTENSITY] > 0]
    data_df = data_df[data_df["Condition"] != "Empty"]

    # Filter peptides with less amino acids than min_aa (default: 7)
    data_df = data_df[
        data_df.apply(lambda x: len(x[PEPTIDE_CANONICAL]) >= min_aa, axis=1)
    ]
    data_df[PROTEIN_NAME] = data_df[PROTEIN_NAME].apply(parse_uniprot_accession)
    data_df[STUDY_ID] = data_df[SAMPLE_ID].apply(get_study_accession)
    if FRACTION not in data_df.columns:
        data_df[FRACTION] = 1
        data_df = data_df[
            [
                PROTEIN_NAME,
                PEPTIDE_SEQUENCE,
                PEPTIDE_CANONICAL,
                PEPTIDE_CHARGE,
                INTENSITY,
                REFERENCE,
                CONDITION,
                RUN,
                BIOREPLICATE,
                FRACTION,
                FRAGMENT_ION,
                ISOTOPE_LABEL_TYPE,
                STUDY_ID,
                SAMPLE_ID,
            ]
        ]
    data_df[CONDITION] = pd.Categorical(data_df[CONDITION])
    data_df[STUDY_ID] = pd.Categorical(data_df[STUDY_ID])
    data_df[SAMPLE_ID] = pd.Categorical(data_df[SAMPLE_ID])

    return data_df


def get_peptidoform_normalize_intensities(
    dataset: pd.DataFrame, higher_intensity: bool = True
) -> pd.DataFrame:
    """
    Select the best peptidoform for the same sample and the same replicates. A peptidoform is the combination of
    a (PeptideSequence + Modifications) + Charge state.
    :param dataset: dataset including all properties
    :param higher_intensity: select based on normalize intensity, if false based on best scored peptide
    :return:
    """
    dataset.dropna(subset=[NORM_INTENSITY], inplace=True)
    if higher_intensity:
        dataset = dataset.loc[
            dataset.groupby(
                [PEPTIDE_SEQUENCE, PEPTIDE_CHARGE, SAMPLE_ID, CONDITION, BIOREPLICATE],
                observed=True,
            )[NORM_INTENSITY].idxmax()
        ]
    else:
        dataset = dataset.loc[
            dataset.groupby(
                [PEPTIDE_SEQUENCE, PEPTIDE_CHARGE, SAMPLE_ID, CONDITION, BIOREPLICATE],
                observed=True,
            )[SEARCH_ENGINE].idxmax()
        ]
    dataset.reset_index(drop=True, inplace=True)
    return dataset


def sum_peptidoform_intensities(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Sum the peptidoform intensities for all peptidofrom across replicates of the same sample.
    :param dataset: Dataframe to be analyzed
    :return: dataframe with the intensities
    """
    dataset.dropna(subset=[NORM_INTENSITY], inplace=True)
    dataset = dataset[
        [
            PROTEIN_NAME,
            PEPTIDE_CANONICAL,
            SAMPLE_ID,
            BIOREPLICATE,
            CONDITION,
            NORM_INTENSITY,
        ]
    ]
    dataset.loc[:, "NormIntensity"] = dataset.groupby(
        [PROTEIN_NAME, PEPTIDE_CANONICAL, SAMPLE_ID, BIOREPLICATE, CONDITION],
        observed=True,
    )[NORM_INTENSITY].transform("sum")
    dataset = dataset.drop_duplicates()
    dataset.reset_index(inplace=True, drop=True)
    return dataset


class Feature:

    def __init__(self, database_path: str):
        if os.path.exists(database_path):
            self.parquet_db = duckdb.connect()
            self.parquet_db = self.parquet_db.execute(
                "CREATE VIEW parquet_db AS SELECT * FROM parquet_scan('{}')".format(
                    database_path
                )
            )
            self.samples = self.get_unique_samples()
        else:
            raise FileNotFoundError(f"the file {database_path} does not exist.")

    @property
    def experimental_inference(self) -> tuple:
        self.labels = self.get_unique_labels()
        self.label, self.choice = get_label(self.labels)
        self.technical_repetitions = self.get_unique_tec_reps()
        return len(self.technical_repetitions), self.label, self.samples, self.choice

    @property
    def low_frequency_peptides(self, percentage=0.2) -> tuple:
        """Return peptides with low frequency"""
        f_table = self.parquet_db.sql(
            """
                select "sequence","protein_accessions",COUNT(DISTINCT sample_accession) as "count" from parquet_db
                GROUP BY "sequence","protein_accessions"
                """
        ).df()
        try:
            f_table["protein_accessions"] = f_table["protein_accessions"].apply(
                lambda x: x[0].split("|")[1]
            )
        except IndexError:
            f_table["protein_accessions"] = f_table["protein_accessions"].apply(
                lambda x: x[0]
            )
        except Exception as e:
            print(e)
            exit(
                "Some errors occurred when parsing protein_accessions column in feature parquet!"
            )
        f_table.set_index(["sequence", "protein_accessions"], inplace=True)
        f_table.drop(
            f_table[f_table["count"] >= (percentage * len(self.samples))].index,
            inplace=True,
        )
        f_table.reset_index(inplace=True)
        return tuple(zip(f_table["protein_accessions"], f_table["sequence"]))

    @staticmethod
    def csv2parquet(csv):
        parquet_path = os.path.splitext(csv)[0] + ".parquet"
        duckdb.read_csv(csv).to_parquet(parquet_path)

    @staticmethod
    def get_label(labels: list) -> (str, dict):
        """Return label type and choice dict according to labels list.

        :param labels: Labels from SDRF.
        :return: Tuple contains label type and choice dict.
        """
        choice = None
        if len(labels) == 1 and (
            "LABEL FREE" in ",".join(labels) or "label free" in ",".join(labels)
        ):
            label = "LFQ"
        elif "TMT" in ",".join(labels) or "tmt" in ",".join(labels):
            if (
                len(labels) > 11
                or "TMT134N" in labels
                or "TMT133C" in labels
                or "TMT133N" in labels
                or "TMT132C" in labels
                or "TMT132N" in labels
            ):
                choice = TMT16plex
            elif len(labels) == 11 or "TMT131C" in labels:
                choice = TMT11plex
            elif len(labels) > 6:
                choice = TMT10plex
            else:
                choice = TMT6plex
            label = "TMT"
        elif "ITRAQ" in ",".join(labels) or "itraq" in ",".join(labels):
            if len(labels) > 4:
                choice = ITRAQ8plex
            else:
                choice = ITRAQ4plex
            label = "ITRAQ"
        else:
            exit("Warning: Only support label free, TMT and ITRAQ experiment!")
        return label, choice

    def get_report_from_database(self, samples: list):
        """
        This function loads the report from the duckdb database for a group of ms_runs.
        :param runs: A list of ms_runs
        :return: The report
        """
        database = self.parquet_db.sql(
            """SELECT * FROM parquet_db WHERE sample_accession IN {}""".format(
                tuple(samples)
            )
        )
        report = database.df()
        return report

    def iter_samples(self, file_num: int = 20):
        """
        :params file_num: The number of files being processed at the same time(default 20)
        :yield: _description_
        """
        ref_list = [
            self.samples[i : i + file_num]
            for i in range(0, len(self.samples), file_num)
        ]
        for refs in ref_list:
            batch_df = self.get_report_from_database(refs)
            yield refs, batch_df

    def get_unique_samples(self):
        """
        return: A list of samples.
        """
        unique = self.parquet_db.sql(
            f"SELECT DISTINCT sample_accession FROM parquet_db"
        ).df()
        return unique["sample_accession"].tolist()

    def get_unique_labels(self):
        """
        return: A list of labels.
        """
        unique = self.parquet_db.sql(
            f"SELECT DISTINCT isotope_label_type FROM parquet_db"
        ).df()
        return unique["isotope_label_type"].tolist()

    def get_unique_tec_reps(self):
        """
        return: A list of labels.
        """
        unique = self.parquet_db.sql(f"SELECT DISTINCT run FROM parquet_db").df()
        try:
            unique["run"] = unique["run"].astype("int")
        except ValueError:
            unique["run"] = unique["run"].str.split("_")[1]
            unique["run"] = unique["run"].astype("int")
        else:
            exit(
                f"Some errors occurred when getting technical repetitions: {Exception}"
            )

        return unique["run"].tolist()
