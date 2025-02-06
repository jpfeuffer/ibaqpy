import logging
from typing import Union

import pandas as pd

from ibaqpy.ibaq.ibaqpy_commons import (
    IBAQ,
    IBAQ_NORMALIZED,
    IBAQ_PPB,
    IBAQ_LOG,
    TPA,
    COPYNUMBER,
    PROTEIN_NAME,
    SAMPLE_ID,
)


def remove_samples_low_protein_number(
    ibaq_df: pd.DataFrame, min_protein_num: int
) -> pd.DataFrame:
    """
    This functions takes an ibaq Dataframe with the following columns:
    - ProteinName
    - SampleID
    - Condition
    - iBAQ
    ...
    ...
    ...
    - iBAQLog

    and removes samples with low protein number. The minimum number of proteins in a sample is defined by the min_protein_num parameter.

    :param ibaq_df: pd.DataFrame
    :param min_protein_num: int
    :return: pd.DataFrame
    """

    protein_num = ibaq_df.groupby(SAMPLE_ID)[PROTEIN_NAME].nunique()

    # Get the samples with more than min_protein_num proteins
    samples_to_keep = protein_num[protein_num >= min_protein_num].index
    samples_to_remove = protein_num[protein_num < min_protein_num].index

    logging.info(
        "The number of samples with number of proteins lower than {} is {}".format(
            min_protein_num, len(samples_to_remove)
        )
    )

    # Filter the samples
    ibaq_df = ibaq_df[ibaq_df["SampleID"].isin(samples_to_keep)]
    return ibaq_df


def remove_missing_values(
    ibaq_df: pd.DataFrame,
    missingness_percentage: float = 30,
    expression_column: str = IBAQ,
) -> pd.DataFrame:
    """
    This functions takes an ibaq Dataframe with the following columns:
    - ProteinName
    - SampleID
    - Condition
    - iBAQ
    ...
    ...
    ...
    - iBAQLog

    and removes the samples that for all the proteins has missing values for a given expression column
    and remove the samples that have a percentage of missing values higher than missingness_percentage.

    :param ibaq_df: pd.DataFrame
    :param missingness_percentage: float
    :param expression_column: str The expression column to consider missigness percentage
    :return: pd.DataFrame
    """

    # Ensure the input is a DataFrame
    if not isinstance(ibaq_df, pd.DataFrame):
        raise ValueError("The input ibaq_df must be a pandas DataFrame.")

    if expression_column not in ibaq_df.columns:
        raise ValueError(
            f"The expression column '{expression_column}' is not in the DataFrame."
        )

    # Initial number of samples
    initial_sample_count = ibaq_df["SampleID"].nunique()
    logging.info(f"Initial number of samples: {initial_sample_count}")

    # Create a pivot table to organize data by ProteinName and SampleID
    pivot_df = ibaq_df.pivot_table(
        index=PROTEIN_NAME, columns=SAMPLE_ID, values=expression_column
    )

    # Remove samples where all proteins have missing values
    non_missing_samples = pivot_df.columns[pivot_df.notna().any(axis=0)]

    # Compute missingness percentage per sample
    missingness = pivot_df[non_missing_samples].isna().sum() / len(pivot_df) * 100

    # Filter samples based on the missingness percentage threshold
    valid_samples = missingness[missingness <= missingness_percentage].index

    # Filter the original DataFrame for valid samples
    filtered_df = ibaq_df[ibaq_df[SAMPLE_ID].isin(valid_samples)]

    # Final number of samples
    final_sample_count = filtered_df[SAMPLE_ID].nunique()
    print(f"Final number of samples: {final_sample_count}")

    removed_sample_count = initial_sample_count - final_sample_count
    print(f"Number of samples removed: {removed_sample_count}")

    return filtered_df


def describe_expression_metrics(ibaq_df: pd.DataFrame) -> pd.DataFrame:
    """
    This function takes an ibaq Dataframe with the following columns:
    - ProteinName
    - SampleID
    - Condition
    - iBAQ
    ...
    ...
    ...
    - iBAQLog

    and return a dataframe for each expression column with the following metrics:
    - standard deviation across samples
    - mean across samples
    - minimum across samples
    - 25th percentile across samples
    - median across samples
    - 75th percentile across samples
    - maximum across samples

    :param ibaq_df: pd.DataFrame
    :return: pd.DataFrame
    """

    possible_expression_values = [
        IBAQ,
        IBAQ_NORMALIZED,
        IBAQ_LOG,
        IBAQ_PPB,
        TPA,
        COPYNUMBER,
    ]

    # Define the expression columns
    expression_columns = [
        col for col in ibaq_df.columns if col in possible_expression_values
    ]

    # Get the metrics
    metrics = ibaq_df.groupby(SAMPLE_ID)[expression_columns].describe()
    return metrics


def pivot_wider(
    df: pd.DataFrame,
    row_name: str,
    col_name: str,
    values: str,
    fillna: Union[int, float, bool] = False,
) -> pd.DataFrame:
    """
    Create a matrix from a DataFrame given the row, column, and value columns.

    Parameters:
    df (pd.DataFrame): The input DataFrame in long format.
    row_name (str): The column name to use as row labels (e.g., sample_ids).
    col_name (str): The column name to use as column labels (e.g., protein_names).
    values (str): The column name to use as cell values (e.g., expression_values).
    fillna (Optional[Union[bool, int, float]]): Value to fill NaN. If True, fill NaN with 0.
                                                If False or None, leave NaN as is.
                                                If a number is provided, use that value.

    Returns:
    pd.DataFrame: A pivot table (matrix) with specified rows, columns, and values.

    Examples:
       df_matrix =  pivot_wider(combined_df,
                    row_name='SampleID',
                    col_name='ProteinName',
                    values='Ibaq',
                    fillna=False)
    """
    # Check if the provided columns exist in the DataFrame
    missing_columns = {row_name, col_name, values} - set(df.columns)
    if missing_columns:
        raise ValueError(f"Columns {missing_columns} not found in the DataFrame.")

    # Check for duplicate combinations
    duplicates = df.groupby([row_name, col_name]).size()
    if (duplicates > 1).any():
        raise ValueError(
            f"Found duplicate combinations of {row_name} and {col_name}. "
            "Use an aggregation function to handle duplicates."
        )

    # Use pivot_table to create the matrix
    matrix = df.pivot_table(
        index=row_name, columns=col_name, values=values, aggfunc="first"
    )

    # Simplified NaN handling
    if fillna is True:  # Fill with 0 if True
        matrix = matrix.fillna(0)
    elif fillna not in [None, False]:  # Fill if a specific value is provided
        matrix = matrix.fillna(fillna)

    return matrix


def pivot_longer(
    df: pd.DataFrame, row_name: str, col_name: str, values: str
) -> pd.DataFrame:
    # Validate input DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    # Validate row_name exists in DataFrame
    if row_name not in df.columns:
        raise ValueError(f"Row name '{row_name}' not found in DataFrame")

    # Reset the index to convert the row labels to a column
    matrix_reset = df.reset_index()

    # Use pd.melt to convert the wide-format DataFrame to long-format
    long_df = pd.melt(
        matrix_reset, id_vars=[row_name], var_name=col_name, value_name=values
    )

    # Remove rows with missing values if any
    if long_df[values].isna().any():
        logging.warning(
            f"Found {long_df[values].isna().sum()} missing values in the result"
        )

    return long_df
