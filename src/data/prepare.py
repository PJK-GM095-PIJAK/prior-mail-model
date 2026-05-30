"""End-to-end data preparation for the priority classifier.

This is the entry point behind ``make data``. It chains the data-layer pieces:

    load (loaders) -> pool both upstream splits -> clean + build model input
    (preprocess) -> stratified 70/15/15 split (splits) -> save to data/processed/

We re-split from scratch (ML_PIPELINE.md §2): the upstream train/test boundary is
discarded so we get a fixed, stratified, reproducible split *with* a validation
set (the protocol needs one for early stopping on macro F1).

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

from src.data.loaders import load_priority_dataset
from src.data.preprocess import build_priority_input
from src.data.splits import stratified_split

logger = logging.getLogger(__name__)

DEFAULT_SEED = 42
DEFAULT_OUTPUT = Path("data/processed/priority")


def prepare_priority(seed: int = DEFAULT_SEED, output_dir: Path = DEFAULT_OUTPUT):
    """Run the full priority data pipeline and save the result. Returns the splits."""
    from datasets import concatenate_datasets

    ds = load_priority_dataset()

    # Re-split from scratch: pool every available example into one dataset.
    pooled = concatenate_datasets([ds[split] for split in ds])
    logger.info("Pooled %d examples from splits %s", pooled.num_rows, list(ds))

    def _add_model_input(row: dict) -> dict:
        return {"model_input": build_priority_input(row.get("subject"), row.get("body"))}

    pooled = pooled.map(_add_model_input)

    splits = stratified_split(pooled, seed=seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    splits.save_to_disk(str(output_dir))
    logger.info("Saved processed priority dataset to %s", output_dir)
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
