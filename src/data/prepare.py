"""End-to-end data preparation for the priority classifier.

This is the entry point behind ``make data``. It chains the data-layer pieces:

    load (loaders) -> pool both upstream splits -> clean + build model input
    (preprocess) -> stratified 70/15/15 split (splits) -> save to data/processed/

We re-split from scratch (ML_PIPELINE.md §2): the upstream train/test boundary is
discarded so we get a fixed, stratified, reproducible split *with* a validation
set (the protocol needs one for early stopping on macro F1).

The internal Indonesian labeled set (§7) is handled deliberately to AVOID
LEAKAGE: only the public set is pooled + re-split into train/val/test, and
internal records are appended to the resulting **train** split afterwards. They
never enter validation or test, so the held-out benchmark stays clean and
comparable across runs.

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

from src.data.labeled import load_labeled_dataset
from src.data.loaders import load_priority_dataset
from src.data.preprocess import build_priority_input
from src.data.splits import stratified_split

logger = logging.getLogger(__name__)

DEFAULT_SEED = 42
DEFAULT_OUTPUT = Path("data/processed/priority")
SPLIT_COLUMNS = ["subject", "body", "priority", "labels"]


def _add_model_input(row: dict) -> dict:
    return {"model_input": build_priority_input(row.get("subject"), row.get("body"))}


def append_internal_to_train(splits, internal):
    """Append the internal set to the TRAIN split only; leave val/test untouched.

    Pulled out as a pure function so the leak-safety guarantee is unit-testable
    without loading the public dataset. ``internal`` may be ``None`` (no-op).
    """
    if internal is None:
        return splits
    from datasets import concatenate_datasets

    train = splits["train"].select_columns(SPLIT_COLUMNS)
    internal = internal.select_columns(SPLIT_COLUMNS)
    # Align Arrow feature types (e.g. string vs large_string) so concat succeeds.
    internal = internal.cast(train.features)
    splits["train"] = concatenate_datasets([train, internal])
    return splits


def prepare_priority(seed: int = DEFAULT_SEED, output_dir: Path = DEFAULT_OUTPUT):
    """Run the full priority data pipeline and save the result. Returns the splits.

    Internal labeled records (§7) are appended to the TRAIN split only — they are
    never pooled into the public re-split, so validation/test stay leak-free.
    """
    from datasets import concatenate_datasets

    # Public set only — internal data is excluded here so it can't leak into the
    # re-split that produces validation/test.
    ds = load_priority_dataset(include_internal=False)

    # Re-split the public data from scratch into train/val/test.
    pooled = concatenate_datasets([ds[split].select_columns(SPLIT_COLUMNS) for split in ds])
    logger.info("Pooled %d public examples from splits %s", pooled.num_rows, list(ds))
    splits = stratified_split(pooled, seed=seed)

    # Append the internal labeled set to TRAIN only (domain adaptation, §7).
    internal = load_labeled_dataset()
    if internal is not None:
        logger.info("Appending %d internal records to train only.", internal.num_rows)
    splits = append_internal_to_train(splits, internal)

    # Build the cleaned model_input on every split AFTER assembly.
    splits = splits.map(_add_model_input)

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
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="fixed split seed")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="output directory")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    prepare_priority(seed=args.seed, output_dir=args.output)


if __name__ == "__main__":
    _main()
