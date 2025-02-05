import os
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


PROTEIN_NAME = "ProteinName"
PEPTIDE_SEQUENCE = "PeptideSequence"
PEPTIDE_CANONICAL = "PeptideCanonical"
PEPTIDE_CHARGE = "PrecursorCharge"
CHANNEL = "Channel"
MIXTRUE = "Mixture"
TECHREPMIXTURE = "TechRepMixture"
CONDITION = "Condition"
BIOREPLICATE = "BioReplicate"
TECHREPLICATE = "TechReplicate"
RUN = "Run"
FRACTION = "Fraction"
INTENSITY = "Intensity"
NORM_INTENSITY = "NormIntensity"
REFERENCE = "Reference"
SAMPLE_ID = "SampleID"
SAMPLE_ID_REGEX = r'^[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*$'
SEARCH_ENGINE = "searchScore"
SCAN = "Scan"
MBR = "MatchBetweenRuns"
IBAQ = "Ibaq"
IBAQ_NORMALIZED = "IbaqNorm"
IBAQ_LOG = "IbaqLog"
IBAQ_PPB = "IbaqPpb"
TPA = "TPA"
MOLECULARWEIGHT = "MolecularWeight"
COPYNUMBER = "CopyNumber"
CONCENTRATION_NM = "Concentration[nM]"
WEIGHT_NG = "Weight[ng]"
MOLES_NMOL = "Moles[nmol]"
GLOBALMEDIAN = "globalMedian"
CONDITIONMEDIAN = "conditionMedian"


PARQUET_COLUMNS = [
    "pg_accessions",
    "peptidoform",
    "sequence",
    "precursor_charge",
    "channel",
    "condition",
    "biological_replicate",
    "run",
    "fraction",
    "intensity",
    "reference_file_name",
    "sample_accession",
]


parquet_map = {
    "pg_accessions": PROTEIN_NAME,
    "peptidoform": PEPTIDE_SEQUENCE,
    "sequence": PEPTIDE_CANONICAL,
    "precursor_charge": PEPTIDE_CHARGE,
    "channel": CHANNEL,
    "condition": CONDITION,
    "biological_replicate": BIOREPLICATE,
    "run": RUN,
    "fraction": FRACTION,
    "intensity": INTENSITY,
    "reference_file_name": REFERENCE,
    "sample_accession": SAMPLE_ID,
}


def get_accession(identifier: str) -> str:
    """
    Get protein accession from the identifier  (e.g. sp|P12345|PROT_NAME)
    :param identifier: Protein identifier
    :return: Protein accession
    """
    identifier_lst = identifier.split("|")
    if len(identifier_lst) == 1:
        return identifier_lst[0]
    else:
        return identifier_lst[1]


def plot_distributions(
    dataset: pd.DataFrame,
    field: str,
    class_field: str,
    title: str = "",
    log2: bool = True,
    width: float = 10,
) -> matplotlib.pyplot:
    """
    Print the quantile plot for the dataset
    :param dataset: DataFrame
    :param field: Field that would be use in the dataframe to plot the quantile
    :param class_field: Field to group the quantile into classes
    :param title: Title of the box plot
    :param log2: Log the intensity values
    :param width: size of the plot
    :return:
    """
    pd.set_option("mode.chained_assignment", None)
    normalize = dataset[[field, class_field]].reset_index(drop=True)
    if log2:
        normalize[field] = np.log2(normalize[field])
    normalize.dropna(subset=[field], inplace=True)
    plt.figure(dpi=500, figsize=(width, 8))
    fig = sns.kdeplot(data=normalize, x=field, hue=class_field, palette="Paired", linewidth=2)
    sns.despine(ax=fig, top=True, right=True)
    plt.title(title)
    pd.set_option("mode.chained_assignment", "warn")

    return plt.gcf()


def plot_box_plot(
    dataset: pd.DataFrame,
    field: str,
    class_field: str,
    log2: bool = False,
    width: float = 10,
    rotation: int = 30,
    title: str = "",
    violin: bool = False,
) -> matplotlib.pyplot:
    """
    Plot a box plot of two values field and classes field
    :param violin: Also add violin on top of box plot
    :param dataset: Dataframe with peptide intensities
    :param field: Intensity field
    :param class_field: class to group the peptides
    :param log2: transform peptide intensities to log scale
    :param width: size of the plot
    :param rotation: rotation of the x-axis
    :param title: Title of the box plot
    :return:
    """
    pd.set_option("mode.chained_assignment", None)
    normalized = dataset[[field, class_field]]
    np.seterr(divide="ignore")
    plt.figure(figsize=(width, 14))
    if log2:
        normalized[field] = np.log2(normalized[field])

    if violin:
        chart = sns.violinplot(
            x=class_field,
            y=field,
            data=normalized,
            boxprops=dict(alpha=0.3),
            palette="muted",
        )
    else:
        chart = sns.boxplot(
            x=class_field,
            y=field,
            data=normalized,
            boxprops=dict(alpha=0.3),
            palette="muted",
        )

    chart.set(title=title)
    chart.set_xticklabels(chart.get_xticklabels(), rotation=rotation, ha="right")
    pd.set_option("mode.chained_assignment", "warn")

    return plt.gcf()


# Functions needed by Combiner
def load_sdrf(sdrf_path: str) -> pd.DataFrame:
    """
    Load sdrf TSV as a dataframe.
    :param sdrf_path: Path to SDRF TSV.
    :return:
    """
    if not os.path.exists(sdrf_path):
        raise FileNotFoundError(f"{sdrf_path} does not exist!")
    sdrf_df = pd.read_csv(sdrf_path, sep="\t")
    sdrf_df.columns = [col.lower() for col in sdrf_df.columns]
    return sdrf_df


def load_feature(feature_path: str) -> pd.DataFrame:
    """
    Load feature file as a dataframe.
    :param feature_path: Path to feature file.
    :return:
    """
    suffix = os.path.splitext(feature_path)[1][1:]
    if suffix == "parquet":
        return pd.read_parquet(feature_path)
    elif suffix == "csv":
        return pd.read_csv(feature_path)
    else:
        raise SystemExit(
            f"{suffix} is not allowed as input, please provide msstats_in or feature parquet."
        )


def is_parquet(path: str) -> bool:
    try:
        with open(path, 'rb') as fh:
            header = fh.read(4)
        return header == b"PAR1"
    except IOError:
        return False
