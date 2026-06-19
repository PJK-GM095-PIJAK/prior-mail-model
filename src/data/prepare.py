"""End-to-end data preparation for the priority and phishing classifiers.

This is the entry point behind ``make data`` and ``make data-phishing``. It
chains the data-layer pieces:

    load (loaders) -> build model input (preprocess) -> save to data/processed/

Priority: the ``insanar/prior-mail-priority`` dataset already ships fixed,
versioned ``train/validation/test`` splits and direct 4-class labels, so this
step does NOT re-split: it honors the published splits as-is.

Phishing: ``ealvaradob/phishing-dataset`` (texts subset) + Enron legit emails
are combined, deduplicated, and stratified-split 70/15/15.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.data.loaders import HF_PRIORITY_CONFIG, load_priority_dataset
from src.data.preprocess import build_priority_input
from src.data.splits import stratified_split

logger = logging.getLogger(__name__)

DEFAULT_SEED = 42
DEFAULT_OUTPUT = Path("data/processed/priority")
KEEP_COLUMNS = ["subject", "body", "priority", "labels", "model_input"]

PHISHING_DEFAULT_OUTPUT = Path("data/processed/phishing")
PHISHING_SPLIT_COLUMNS = ["sender_email", "subject", "body", "phishing", "labels"]


def _add_model_input(row: dict) -> dict:
    return {"model_input": build_priority_input(row.get("subject"), row.get("body"))}


def prepare_priority(config: str = HF_PRIORITY_CONFIG, output_dir: Path = DEFAULT_OUTPUT):
    """Load the priority dataset, build ``model_input``, and save it. Returns the splits.

    Honors the dataset's published ``train/validation/test`` splits (no re-split).

    Args:
        config: HF dataset config to build from (default ``v4``).
        output_dir: where to ``save_to_disk`` the processed ``DatasetDict``.
    """
    splits = load_priority_dataset(config=config)

    splits = splits.map(_add_model_input)
    splits = splits.remove_columns(
        [c for c in splits["train"].column_names if c not in KEEP_COLUMNS]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    splits.save_to_disk(str(output_dir))
    logger.info(
        "Saved processed priority dataset to %s (train=%d val=%d test=%d)",
        output_dir,
        splits["train"].num_rows,
        splits["validation"].num_rows,
        splits["test"].num_rows,
    )
    return splits


def prepare_phishing(
    seed: int = DEFAULT_SEED,
    output_dir: Path = PHISHING_DEFAULT_OUTPUT,
    enron_csv_path: str | Path = "emails.csv",
    legit_sample_size: int = 20_000,
    augmentation_size: int = 600,
    nazario_csv_path: str | Path = "Nazario.csv",
    use_nazario: bool = False,
    nazario_sample_size: int | None = None,
):
    """Run the full phishing data pipeline and save the result.

    Chains: load (``load_phishing_dataset``) -> build model_input
    (``build_phishing_input``) -> stratified 70/15/15 split -> save to disk.

    Args:
        seed: split seed — record in the training config + wandb run.
        output_dir: where to save the processed DatasetDict.
        enron_csv_path: path to the Enron legit email CSV.
        legit_sample_size: how many Enron rows to sample (negative class).
        augmentation_size: v2.1 fix B1 — synthetic rows per class to add (covers
            BEC / fake-invoice / credential tactics the corpora miss). 0 disables.
        nazario_csv_path: path to the Nazario phishing CSV (real phishing corpus).
        use_nazario: v2.2 — add the real Nazario phishing corpus to the phishing
            class (real, header-bearing phishing beyond the synthetic templates).
        nazario_sample_size: optional cap on Nazario rows (random, seeded).
    """
    from src.data.loaders import load_phishing_dataset
    from src.data.preprocess import build_phishing_input

    ds = load_phishing_dataset(
        enron_csv_path=enron_csv_path,
        legit_sample_size=legit_sample_size,
        seed=seed,
        augmentation_size=augmentation_size,
        nazario_csv_path=nazario_csv_path,
        use_nazario=use_nazario,
        nazario_sample_size=nazario_sample_size,
    )

    pooled = ds["train"].select_columns(PHISHING_SPLIT_COLUMNS)
    logger.info("Pooled %d phishing examples for stratified split", pooled.num_rows)

    splits = stratified_split(pooled, seed=seed)

    def _add_phishing_model_input(row: dict) -> dict:
        return {
            "model_input": build_phishing_input(
                row.get("sender_email"), row.get("subject"), row.get("body")
            )
        }

    splits = splits.map(_add_phishing_model_input)
    # Drop rows whose body-only input is empty (a few source emails have no body)
    # — a blank input is a degenerate training example.
    before = {k: splits[k].num_rows for k in splits}
    splits = splits.filter(lambda r: bool(r["model_input"].strip()))
    dropped = sum(before[k] - splits[k].num_rows for k in splits)
    if dropped:
        logger.info("Dropped %d rows with empty model_input", dropped)

    output_dir.mkdir(parents=True, exist_ok=True)
    splits.save_to_disk(str(output_dir))
    logger.info(
        "Saved processed phishing dataset to %s (train=%d val=%d test=%d)",
        output_dir,
        splits["train"].num_rows,
        splits["validation"].num_rows,
        splits["test"].num_rows,
    )
    return splits


def _main() -> None:
    parser = argparse.ArgumentParser(description="Prepare priority or phishing dataset (make data)")
    parser.add_argument("--output", type=Path, default=None, help="output directory override")
    parser.add_argument(
        "--subset", default=HF_PRIORITY_CONFIG, help="HF dataset config (default: v4)"
    )
    parser.add_argument(
        "--phishing", action="store_true",
        help="prepare the phishing dataset instead of the priority dataset",
    )
    parser.add_argument(
        "--enron-csv", default="emails.csv",
        help="path to the Enron legit email CSV (phishing mode only)",
    )
    parser.add_argument(
        "--legit-sample-size", type=int, default=20_000,
        help="how many Enron rows to sample as legit negatives (phishing mode only)",
    )
    parser.add_argument(
        "--augmentation-size", type=int, default=600,
        help="synthetic rows per class to add (phishing mode only; 0 disables, v2.1 fix B1)",
    )
    parser.add_argument(
        "--nazario-csv", default="Nazario.csv",
        help="path to the Nazario phishing CSV (phishing mode only)",
    )
    parser.add_argument(
        "--use-nazario", action="store_true",
        help="add the real Nazario phishing corpus to the phishing class (v2.2)",
    )
    parser.add_argument(
        "--nazario-sample-size", type=int, default=None,
        help="optional cap on Nazario rows sampled (phishing mode only)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.phishing:
        out = args.output if args.output else PHISHING_DEFAULT_OUTPUT
        prepare_phishing(
            seed=DEFAULT_SEED,
            output_dir=out,
            enron_csv_path=args.enron_csv,
            legit_sample_size=args.legit_sample_size,
            augmentation_size=args.augmentation_size,
            nazario_csv_path=args.nazario_csv,
            use_nazario=args.use_nazario,
            nazario_sample_size=args.nazario_sample_size,
        )
    else:
        out = args.output if args.output else DEFAULT_OUTPUT
        prepare_priority(config=args.subset, output_dir=out)


if __name__ == "__main__":
    _main()
