import logging
from pathlib import Path
import pytest
import anndata

import pandas as pd

from ibaqpy.commands.correct_batches import run_batch_correction
from ibaqpy.ibaq.ibaqpy_commons import SAMPLE_ID, PROTEIN_NAME, IBAQ, IBAQ_BEC

TESTS_DIR = Path(__file__).parent


def test_correct_batches():
    # Test valid case
    args = {
        "folder": TESTS_DIR / "ibaq-raw-hela",
        "pattern": "*ibaq.tsv",
        "comment": "#",
        "sep": "\t",
        "output": TESTS_DIR / "example/ibaq_corrected_combined.tsv",
        "sample_id_column": SAMPLE_ID,
        "protein_id_column": PROTEIN_NAME,
        "ibaq_raw_column": IBAQ,
        "ibaq_corrected_column": IBAQ_BEC,
        "export_anndata": True,
    }
    logging.debug("Arguments for run_batch_correction: %s", args)
    run_batch_correction(**args)

    # Assert the output file is created and not empty
    output_path = Path(args["output"])
    assert output_path.exists(), f"Expected output file {output_path} was not created."
    df = pd.read_csv(output_path, sep=args["sep"])
    assert not df.empty, "The corrected output file is empty."

    # Assert the AnnData object is created
    adata_path = output_path.with_suffix(".h5ad")
    assert adata_path.exists(), f"Expected AnnData file {adata_path} was not created."

    # Read the AnnData object and check shape and layers
    adata = anndata.read_h5ad(adata_path)
    print(adata)
    assert adata.shape == (46, 3476)
    assert adata.layers[IBAQ_BEC].shape == (46, 3476)

    # Test invalid sample IDs
    with pytest.raises(ValueError):
        args["folder"] = TESTS_DIR / "invalid-samples"
        run_batch_correction(**args)

    # Test missing required columns
    with pytest.raises(ValueError):
        args["folder"] = TESTS_DIR / "ibaq-raw-hela"
        args["sample_id_column"] = "NonexistentColumn"
        run_batch_correction(**args)

    # Test invalid file pattern
    with pytest.raises(ValueError):
        args["pattern"] = "nonexistent*.tsv"
        run_batch_correction(**args)


if __name__ == "__main__":
    test_correct_batches()
