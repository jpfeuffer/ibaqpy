import logging
from pathlib import Path

import pandas as pd

from ibaqpy.ibaq.file_utils import create_anndata, combine_ibaq_tsv_files
from ibaqpy.ibaq.ibaqpy_commons import (
    SAMPLE_ID,
    PROTEIN_NAME,
    IBAQ,
    IBAQ_NORMALIZED,
    IBAQ_LOG,
)

TESTS_DIR = Path(__file__).parent


def test_combine_ibaq_tsv_files():
    ibaq_dir = TESTS_DIR / "ibaq-raw-hela"
    files_pattern = "*ibaq.tsv"
    df_ibaq = combine_ibaq_tsv_files(
        dir_path=str(ibaq_dir), pattern=files_pattern, sep="\t"
    )
    logging.info(df_ibaq.head())
    assert df_ibaq.shape == (83725, 14)


def test_create_anndata():
    df = pd.read_csv(TESTS_DIR / "ibaq-raw-hela/PXD000396.ibaq.tsv", sep="\t")
    obs_col = SAMPLE_ID
    var_col = PROTEIN_NAME
    value_col = IBAQ
    layers = [IBAQ_NORMALIZED, IBAQ_LOG]
    adata = create_anndata(
        df=df,
        obs_col=obs_col,
        var_col=var_col,
        value_col=value_col,
        layer_cols=layers,
        obs_metadata_cols=["Condition"],
        var_metadata_cols=[],
    )
    print(adata)
    assert adata.shape == (12, 3096)
    assert adata.layers[IBAQ_NORMALIZED].shape == (12, 3096)
    assert adata.layers[IBAQ_LOG].shape == (12, 3096)
    assert "HeLa" in adata.obs["Condition"].values

