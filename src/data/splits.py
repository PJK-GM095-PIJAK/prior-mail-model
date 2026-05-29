"""Train/val/test splitting.

ML_PIPELINE.md §2 ("Test Set"): held-out 15% of the internal labeled set + 15%
of the public dataset, **stratified by class**, with a **fixed seed and fixed
split** reused across all runs for comparability.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Fixed split fractions (ML_PIPELINE.md §2). Do not change without re-baselining.
TEST_FRACTION: float = 0.15
VAL_FRACTION: float = 0.15


def stratified_split(dataset, seed: int):
    """Return ``(train, val, test)`` stratified by label with a fixed seed."""
    raise NotImplementedError(
        "stratified_split: implement stratified 70/15/15 split with fixed seed "
        "(ML_PIPELINE.md §2 — must be reused across runs)."
    )
