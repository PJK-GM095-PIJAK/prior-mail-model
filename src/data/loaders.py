"""Dataset loaders for the priority and phishing models.

================================================================================
CLASS MAPPING — external dataset -> PriorMail 4-class scheme
================================================================================
ML_PIPELINE.md §2 ("Class Mapping") requires the mapping from the external
dataset's categories to our 4 classes to be FIXED HERE and documented in this
comment block. The mapping is owned by Insan + Faiz.

Source dataset: ``jason23322/high-accuracy-email-classifier``
Target classes (src.utils.constants.PRIORITY_LABELS): urgent | high | normal | low

    >>> PLACEHOLDER — mapping not yet provided. <<<
    Insan to supply the source-category -> target-class table. Until then,
    ``PRIORITY_CLASS_MAPPING`` is empty and ``map_to_priority`` raises so no run
    can silently train on an unmapped/garbage label set.

Example of the intended shape (illustrative only — NOT the real mapping):
    PRIORITY_CLASS_MAPPING = {
        "spam":       "low",
        "promotions": "low",
        "updates":    "normal",
        "forums":     "normal",
        "primary":    "high",
        # ...
    }
================================================================================
"""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)

# Filled in by Insan (see comment block above). Keep empty until finalized.
PRIORITY_CLASS_MAPPING: dict[str, str] = {}


def map_to_priority(source_label: str) -> str:
    """Map one source-dataset category to a PriorMail priority class.

    Raises:
        NotImplementedError: if the mapping has not been provided yet.
        KeyError: if ``source_label`` is missing from a finalized mapping.
    """
    if not PRIORITY_CLASS_MAPPING:
        raise NotImplementedError(
            "PRIORITY_CLASS_MAPPING is empty — Insan must supply the external-dataset "
            "-> {urgent,high,normal,low} mapping before loading the priority dataset. "
            "See the comment block at the top of src/data/loaders.py."
        )
    return PRIORITY_CLASS_MAPPING[source_label]


def load_priority_dataset(download: bool = False):
    """Load the priority-classifier dataset (HF + internal labeled set).

    Returns a HuggingFace ``DatasetDict`` with mapped 4-class labels. Stub for now.
    """
    raise NotImplementedError("load_priority_dataset: implement once class mapping is set.")


def load_phishing_dataset(download: bool = False):
    """Load the phishing dataset (``ealvaradob/phishing-dataset`` + negatives)."""
    raise NotImplementedError("load_phishing_dataset: not yet implemented.")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Download/prepare datasets")
    parser.add_argument("--download", action="store_true", help="fetch raw data from HF")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logger.info("loaders.py invoked (download=%s) — stub, nothing to do yet.", args.download)


if __name__ == "__main__":
    _main()
