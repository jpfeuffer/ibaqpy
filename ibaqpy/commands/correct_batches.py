import re
from typing import Union

import click
import pandas as pd

from ibaqpy.ibaq.ibaqpy_commons import SAMPLE_ID_REGEX, IBAQ_BEC
from ibaqpy.ibaq.ibaqpy_postprocessing import (
    combine_ibaq_tsv_files,
    pivot_wider,
    pivot_longer,
)
from ibaqpy.ibaq.utils import apply_batch_correction


def is_valid_sample_id(
    samples: Union[str, list, pd.Series], sample_id_pattern: str = SAMPLE_ID_REGEX
) -> bool:
    """
    Validates that all sample IDs in the provided input are composed only of alphanumeric
    characters and hyphen-separated alphanumeric parts.

    Also prints out any sample IDs that fail to match the pattern.

    Args:
        samples (Union[list, str, pd.Series]): A list, single sample ID, or pandas Series containing the sample IDs to validate.
        sample_id_pattern (str): The regex pattern to validate the sample IDs against.

    Returns:
        bool: True if all sample IDs are valid according to the pattern, False otherwise.
    """
    # Compile the regex pattern for a valid sample name.
    sample_pattern = re.compile(sample_id_pattern)

    # Ensure samples is a list for uniform processing
    if isinstance(samples, str):
        samples = [samples]
    elif isinstance(samples, pd.Series):
        samples = samples.tolist()

    # Identify invalid sample names.
    invalid_samples = [sample for sample in samples if not sample_pattern.fullmatch(sample)]

    if invalid_samples:
        print("The following sample IDs are invalid:")
        for invalid_sample in invalid_samples:
            print(f" - {invalid_sample}")
        return False
    return True


def get_batch_id_from_sample_names(samples: list) -> list:
    """
    Extract batch IDs from sample names, assuming batch ID is the first part before the hyphen.

    Args:
        samples (list): List of sample names in the format 'batch-id-suffix'.

    Returns:
        list: List of numeric batch IDs mapped from the extracted batch identifiers.

    Raises:
        ValueError: If sample name format is invalid or batch ID is empty.
    """
    batch_ids = []
    for sample in samples:
        parts = sample.split("-")
        if not parts or not parts[0]:
            raise ValueError(f"Invalid sample name format: {sample}. Expected batch-id prefix.")
        batch_id = parts[0]
        if not re.match(r"^[A-Za-z0-9]+$", batch_id):
            raise ValueError(
                f"Invalid batch ID format: {batch_id}. Expected alphanumeric characters only."
            )
        batch_ids.append(batch_id)
    return pd.factorize(batch_ids)[0]


def run_batch_correction(
    folder: str,
    pattern: str,
    comment: str,
    sep: str,
    output: str,
    sample_id_column: str,
    protein_id_column: str,
    ibaq_column: str,
) -> pd.DataFrame:
    """
    Runs the batch correction on iBAQ values from TSV files.

    Args:
        folder (str): Folder that contains all TSV files with raw iBAQ values.
        pattern (str): Pattern for the TSV files with raw iBAQ values.
        comment (str): Comment character for the TSV files. Lines starting with this character will be ignored.
        sep (str): Separator for the TSV files.
        output (str): Output file name for the combined iBAQ corrected values.
        sample_id_column (str): Sample ID column name.
        protein_id_column (str): Protein ID column name.
        ibaq_column (str): iBAQ column name.
    """
    # Load the data
    try:
        df_ibaq = combine_ibaq_tsv_files(folder, pattern=pattern, comment=comment, sep=sep)
    except Exception as e:
        raise ValueError(f"Failed to load input files: {str(e)}")

    # Reshape the data to wide format
    df_wide = pivot_wider(
        df_ibaq,
        row_name=protein_id_column,
        col_name=sample_id_column,
        values=ibaq_column,
        fillna=True,
    )

    # Validate the sample IDs
    if not is_valid_sample_id(df_wide.columns, SAMPLE_ID_REGEX):
        raise ValueError("Invalid sample IDs found in the data.")

    # Get the batch IDs
    batch_ids = get_batch_id_from_sample_names(df_wide.columns)

    # Run batch correction
    df_corrected = apply_batch_correction(df_wide, list(batch_ids), kwargs={})

    # Convert the data back to long format
    df_corrected = df_corrected.reset_index()
    df_corrected_long = pivot_longer(
        df_corrected,
        row_name=protein_id_column,
        col_name=sample_id_column,
        values=IBAQ_BEC,
    )

    # Add the corrected ibaq values to the original dataframe.
    # Use sample/protein ID keys to merge the dataframes.
    df_ibaq = df_ibaq.merge(
        df_corrected_long, how="left", on=[sample_id_column, protein_id_column]
    )

    # Save the corrected iBAQ values to a file
    if output:
        try:
            df_ibaq.to_csv(output, sep=sep, index=False)
        except Exception as e:
            raise ValueError(f"Failed to save output file: {str(e)}")

    return df_ibaq


@click.command()
@click.option(
    "-f",
    "--folder",
    help="Folder that contains all TSV files with raw iBAQ values",
    required=True,
    default=None,
)
@click.option(
    "-p",
    "--pattern",
    help="Pattern for the TSV files with raw iBAQ values",
    required=True,
    default="*ibaq.tsv",
)
@click.option(
    "--comment",
    help="Comment character for the TSV files. Lines starting with this character will be ignored.",
    required=False,
    default="#",
)
@click.option("--sep", help="Separator for the TSV files", required=False, default="\t")
@click.option(
    "-o",
    "--output",
    help="Output file name for the combined iBAQ corrected values",
    required=True,
)
@click.option(
    "-sid",
    "--sample_id_column",
    help="Sample ID column name",
    required=False,
    default="SampleID",
)
@click.option(
    "-pid",
    "--protein_id_column",
    help="Protein ID column name",
    required=False,
    default="ProteinName",
)
@click.option("-ibaq", "--ibaq_column", help="iBAQ column name", required=False, default="Ibaq")
@click.pass_context
def correct_batches(
    ctx,
    folder: str,
    pattern: str,
    comment: str,
    sep: str,
    output: str,
    sample_id_column: str,
    protein_id_column: str,
    ibaq_column: str,
):
    run_batch_correction(
        folder=folder,
        pattern=pattern,
        comment=comment,
        sep=sep,
        output=output,
        sample_id_column=sample_id_column,
        protein_id_column=protein_id_column,
        ibaq_column=ibaq_column,
    )
