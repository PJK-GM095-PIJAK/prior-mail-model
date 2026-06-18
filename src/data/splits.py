"""Train/val/test splitting.

Used by the phishing pipeline (``prepare_phishing``): the ealvaradob + priority
legit pool is assembled fresh in ``loaders.load_phishing_dataset`` and has no
upstream split boundary, so we carve a stratified, fixed-seed 70/15/15 split here
for comparability across runs.

(The priority classifier does NOT use this — ``insanar/prior-mail-priority`` ships
its own versioned splits, honored as-is in ``prepare_priority``. This module was
collaterally removed in the DistilBERT migration; restored for phishing v2.)
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Fixed split fractions. Do not change without re-baselining.
TEST_FRACTION: float = 0.15
VAL_FRACTION: float = 0.15
# Column to stratify on: the integer label id added by the loaders.
STRATIFY_COLUMN: str = "labels"


def stratified_split(dataset, seed: int, stratify_column: str = STRATIFY_COLUMN):
    """Split a single ``Dataset`` into stratified train/val/test (70/15/15).

    The split is stratified by class with a FIXED seed and reused across all runs,
    so metrics are comparable. Pass a ``Dataset`` that already pools every
    available example.

    Args:
        dataset: a HuggingFace ``Dataset`` with the stratify column present.
        seed: fixed split seed (recorded in the run config + wandb).
        stratify_column: integer class-label column to stratify on.

    Returns:
        A ``DatasetDict`` with ``train`` / ``validation`` / ``test`` splits.
    """
    from datasets import ClassLabel, DatasetDict

    if stratify_column not in dataset.column_names:
        raise KeyError(
            f"stratify column {stratify_column!r} not in dataset columns "
            f"{dataset.column_names}."
        )

    # HF only stratifies on a typed ClassLabel column; our labels are plain ints.
    # Cast a copy so the split is stratified (values are unchanged).
    if not isinstance(dataset.features[stratify_column], ClassLabel):
        num_classes = len(set(dataset[stratify_column]))
        dataset = dataset.cast_column(stratify_column, ClassLabel(num_classes=num_classes))

    # 1) carve out the test set (15% of the whole).
    first = dataset.train_test_split(
        test_size=TEST_FRACTION, seed=seed, stratify_by_column=stratify_column
    )
    test = first["test"]

    # 2) carve val out of the remaining 85% so it ends up at 15% of the whole.
    val_size = VAL_FRACTION / (1.0 - TEST_FRACTION)
    second = first["train"].train_test_split(
        test_size=val_size, seed=seed, stratify_by_column=stratify_column
    )

    splits = DatasetDict({"train": second["train"], "validation": second["test"], "test": test})
    logger.info(
        "Stratified split (seed=%d): train=%d val=%d test=%d",
        seed,
        splits["train"].num_rows,
        splits["validation"].num_rows,
        splits["test"].num_rows,
    )
    return splits
