"""End-to-end data preparation for the priority classifier.

This is the entry point behind ``make data``. It chains the data-layer pieces:

    load (loaders) -> build model input (preprocess) -> save to data/processed/

The priority dataset (``insanar/prior-mail-priority``) already ships fixed,
versioned ``train/validation/test`` splits and direct 4-class labels, so this
step does NOT re-split: it honors the published splits as-is (re-splitting would
mix the dataset's synthetic rows into the held-out test set). All we add is the
cleaned ``model_input`` string the model actually consumes.

Output: a saved HuggingFace ``DatasetDict`` at ``data/processed/priority/`` with
``train`` / ``validation`` / ``test`` splits, each carrying:
  - ``model_input``  the cleaned ``{subject} [SEP] {body}`` string fed to the model
  - ``labels``       integer class id (0..3)
  - ``priority``     the label string (urgent|high|normal|low)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.data.loaders import HF_PRIORITY_CONFIG, load_priority_dataset
from src.data.preprocess import build_priority_input

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path("data/processed/priority")
# Columns kept in the processed set: the model input is built from subject+body;
# subject/body/priority are retained for error analysis and eval reporting.
KEEP_COLUMNS = ["subject", "body", "priority", "labels", "model_input"]


def _add_model_input(row: dict) -> dict:
    return {"model_input": build_priority_input(row.get("subject"), row.get("body"))}


def prepare_priority(config: str = HF_PRIORITY_CONFIG, output_dir: Path = DEFAULT_OUTPUT):
    """Load the priority dataset, build ``model_input``, and save it. Returns the splits.

    Honors the dataset's published ``train/validation/test`` splits (no re-split).

    Args:
        config: HF dataset config to build from (default ``v2``).
        output_dir: where to ``save_to_disk`` the processed ``DatasetDict``.
    """
    splits = load_priority_dataset(config=config)

    # Build the cleaned model_input on every split, then keep only what the
    # trainer/eval need (drops id, label_source, source_category, labeled_at).
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


def _main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the priority dataset (make data)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="output directory")
    parser.add_argument(
        "--subset", default=HF_PRIORITY_CONFIG, help="HF dataset config (default: v2)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    prepare_priority(config=args.subset, output_dir=args.output)


if __name__ == "__main__":
    _main()
