import click

from ibaqpy.ibaq.peptide_normalization import peptide_normalization
from ibaqpy.model.normalization import FeatureNormalizationMethod, PeptideNormalizationMethod


@click.command("features2peptides", short_help="Convert features to parquet file.")
@click.option(
    "-p",
    "--parquet",
    help="Parquet file import generated by quantms.io",
    required=True,
    type=click.Path(exists=True),
)
@click.option(
    "-s", "--sdrf", help="SDRF file import generated by quantms", default=None, type=click.Path()
)
@click.option(
    "--min_aa", help="Minimum number of amino acids to filter peptides", type=int, default=7
)
@click.option(
    "--min_unique",
    help="Minimum number of unique peptides to filter proteins",
    default=2,
    type=int,
)
@click.option(
    "--remove_ids",
    help="Remove specific protein ids from the analysis using a file with one id per line",
    type=click.Path(exists=True),
)
@click.option(
    "--remove_decoy_contaminants",
    help="Remove decoy and contaminants proteins from the analysis",
    is_flag=True,
    default=False,
)
@click.option(
    "--remove_low_frequency_peptides",
    help="Remove peptides that are present in less than 20% of the samples",
    is_flag=True,
    default=False,
)
@click.option(
    "-o",
    "--output",
    help="Peptide intensity file including other all properties for normalization",
    type=click.Path(),
)
@click.option("--skip_normalization", help="Skip normalization step", is_flag=True, default=False)
@click.option(
    "--nmethod",
    help="Normalization method used to normalize feature intensities for tec (options: mean, median, iqr, none)",
    default="median",
    type=click.Choice([f.name.lower() for f in FeatureNormalizationMethod], case_sensitive=False),
)
@click.option(
    "--pnmethod",
    help="Normalization method used to normalize peptides intensities for all samples (options:globalMedian, conditionMedian)",
    default="globalMedian",
    type=click.Choice([p.name.lower() for p in PeptideNormalizationMethod], case_sensitive=False),
)
@click.option(
    "--log2",
    help="Transform to log2 the peptide intensity values before normalization",
    is_flag=True,
)
@click.option(
    "--save_parquet",
    help="Save normalized peptides to parquet",
    is_flag=True,
)
@click.pass_context
def features2parquet(
    ctx,
    parquet: str,
    sdrf: str,
    min_aa: int,
    min_unique: int,
    remove_ids: str,
    remove_decoy_contaminants: bool,
    remove_low_frequency_peptides: bool,
    output: str,
    skip_normalization: bool,
    nmethod: str,
    pnmethod: str,
    log2: bool,
    save_parquet: bool,
) -> None:
    """
    Convert feature data to a parquet file with optional normalization and filtering steps.
    """

    peptide_normalization(
        parquet=parquet,
        sdrf=sdrf,
        min_aa=min_aa,
        min_unique=min_unique,
        remove_ids=remove_ids,
        remove_decoy_contaminants=remove_decoy_contaminants,
        remove_low_frequency_peptides=remove_low_frequency_peptides,
        output=output,
        skip_normalization=skip_normalization,
        nmethod=nmethod,
        pnmethod=pnmethod,
        log2=log2,
        save_parquet=save_parquet,
    )
