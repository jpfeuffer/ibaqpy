import glob
import warnings
from typing import List, Optional

import anndata as an
import pandas as pd

from ibaqpy.ibaq.ibaqpy_postprocessing import pivot_wider


def create_anndata(
    df: pd.DataFrame,
    obs_col: str,
    var_col: str,
    value_col: str,
    layer_cols: Optional[List[str]] = None,
    obs_metadata_cols: Optional[List[str]] = None,
    var_metadata_cols: Optional[List[str]] = None,
) -> an.AnnData:
    """
    Create an AnnData object from a long-format DataFrame.

    Parameters:
        df (pd.DataFrame): Input data in long format.
        obs_col (str): Column name in df representing observation IDs.
        var_col (str): Column name in df representing variable IDs.
        value_col (str): Column name in df representing the main data values.
        layer_cols (Optional[List[str]]): List of column names in df to add as additional layers.
        obs_metadata_cols (Optional[List[str]]): List of column names in df to add as observation metadata.
        var_metadata_cols (Optional[List[str]]): List of column names in df to add as variable metadata.

    Returns:
        anndata.AnnData: The constructed AnnData object.
    """
    if df.empty:
        raise ValueError("Cannot create AnnData object from empty DataFrame")
    # Validate that the required columns exist in the DataFrame.
    required_cols = [obs_col, var_col, value_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"The following required columns are missing from the input DataFrame: {missing}"
        )

    # Pivot the long dataframe to create a wide-format matrix for the main values.
    df_matrix = pivot_wider(
        df, row_name=obs_col, col_name=var_col, values=value_col, fillna=True
    )

    # Create the AnnData object with the main data matrix.
    adata = an.AnnData(
        X=df_matrix.to_numpy(),
        obs=df_matrix.index.to_frame(),
        var=df_matrix.columns.to_frame(),
    )

    def add_metadata(
        metadata_df: pd.DataFrame, key: str, cols: List[str]
    ) -> pd.DataFrame:
        """
        Add metadata columns to a DataFrame by mapping values from the original long dataframe.

        Parameters:
            metadata_df (pd.DataFrame): DataFrame (either adata.obs or adata.var) to update.
            key (str): The column name used as key (obs_col for observations, var_col for variables).
            cols (List[str]): List of metadata columns to add.

        Returns:
            pd.DataFrame: The updated metadata DataFrame.
        """
        for col in cols:
            if col not in df.columns:
                warnings.warn(
                    f"Column '{col}' not found in the input DataFrame. Skipping metadata for '{col}'."
                )
                continue
            # Create a mapping from key to metadata values.
            mapping = df[[key, col]].drop_duplicates().set_index(key)[col]
            metadata_df[col] = metadata_df.index.map(mapping)
        return metadata_df

    # Add observation metadata, if provided.
    if obs_metadata_cols:
        adata.obs = add_metadata(adata.obs, obs_col, obs_metadata_cols)

    # Add variable metadata, if provided.
    if var_metadata_cols:
        adata.var = add_metadata(adata.var, var_col, var_metadata_cols)

    # Add additional layers (if any) using a similar pivot operation.
    if layer_cols:
        for layer_col in layer_cols:
            if layer_col not in df.columns:
                warnings.warn(
                    f"Layer column '{layer_col}' not found in the input DataFrame. Skipping layer '{layer_col}'."
                )
                continue
            df_layer = pivot_wider(
                df, row_name=obs_col, col_name=var_col, values=layer_col, fillna=True
            )
            adata.layers[layer_col] = df_layer.to_numpy()

    return adata


def combine_ibaq_tsv_files(
    dir_path: str, pattern: str = "*", comment: str = "#", sep: str = "\t"
) -> pd.DataFrame:
    """
    Combine multiple TSV files from a directory into a single pandas DataFrame.

    Parameters
    ----------
    dir_path : str
        Directory path containing the TSV files.
    pattern : str, optional
        Pattern to match files in the directory (default is '*').
    comment : str, optional
        Character to indicate the start of a comment line (default is '#').
        It will skip lines starting with this character when reading the TSV files.
    sep : str, optional
        Delimiter to use for reading the TSV files (default is '\t').

    Returns
    -------
    Optional[pd.DataFrame]
        Combined DataFrame containing data from all TSV files, or None if no files match the pattern.

    Examples
    --------
        dir_path = './ibaqpy-research-data/ibaq-hela-raw'
        combined_df = combine_ibaq_tsv_files(dir_path, pattern='*ibaq.tsv', comment='#', sep='\t')
    """
    file_paths = glob.glob(f"{dir_path}/{pattern}")

    if not file_paths:
        raise FileNotFoundError(
            f"No files found in the directory '{dir_path}' matching the pattern '{pattern}'."
        )

    dataframes = []

    first_schema = None
    for file_path in file_paths:
        try:
            # Read the TSV file, skipping lines that start with the comment character
            df = pd.read_csv(file_path, sep=sep, comment=comment)

            # Validate schema consistency
            if first_schema is None:
                first_schema = set(df.columns)
            elif set(df.columns) != first_schema:
                raise ValueError(
                    f"Schema mismatch in file '{file_path}'. "
                    f"Expected columns: {sorted(first_schema)}, "
                    f"got: {sorted(df.columns)}"
                )

            dataframes.append(df)
        except Exception as e:
            raise ValueError(f"Error reading file '{file_path}': {str(e)}")

    # Concatenate all DataFrames
    combined_df = pd.concat(dataframes, ignore_index=True)

    return combined_df
