"""Dataset loaders for the priority and phishing models.

================================================================================
CLASS MAPPING — external dataset -> PriorMail 4-class scheme
================================================================================
ML_PIPELINE.md §2 ("Class Mapping") requires the mapping from the external
dataset's categories to our 4 classes to be FIXED HERE and documented in this
comment block. The mapping is owned by Insan + Faiz.

Source dataset: ``jason23322/high-accuracy-email-classifier``
  - Label column: ``category`` (6 topical classes, ~balanced, ~1800 rows each).
  - Text columns available: ``subject`` + ``body`` (and a combined ``text``).
Target classes (src.utils.constants.PRIORITY_LABELS): urgent | high | normal | low

FINALIZED MAPPING (Insan, 2026-05-30 — pending Faiz review per ML_PIPELINE.md §2):

    category      -> priority   rationale
    ------------     --------   ---------------------------------------------
    verify_code   -> urgent     OTP / 2FA codes; time-sensitive by nature.
    updates       -> high       Broadest bucket; treated as high as a starting
                                prior. NOTE: likely over-prioritizes routine
                                notifications — revisit after error analysis.
    forum         -> normal     Informational discussion mail.
    social_media  -> normal     Social notifications; no action expected.
    promotions    -> low        Marketing / bulk.
    spam          -> low         Unwanted bulk. (Malicious mail is the PHISHING
                                model's job, not a priority class.)

Resulting train distribution: normal 33% / low 33% / urgent 17% / high 17%
(no collapsed class — workable against the §8 per-class-recall ≥ 0.65 gate).

CAVEAT: this is a TOPIC -> urgency proxy, not true urgency. It is a bootstrap
signal only; the internal Indonesian labeled set (ML_PIPELINE.md §7) is what
corrects it for the real domain.
================================================================================
"""

from __future__ import annotations

import argparse
import logging

from src.utils.constants import PRIORITY_LABELS

logger = logging.getLogger(__name__)

# Source ``category`` -> PriorMail priority. See the comment block above for the
# rationale and the per-class distribution. Keys must match the dataset's
# ``category`` values exactly (case-sensitive).
PRIORITY_CLASS_MAPPING: dict[str, str] = {
    "verify_code": "urgent",
    "updates": "high",
    "forum": "normal",
    "social_media": "normal",
    "promotions": "low",
    "spam": "low",
}

# Fail fast at import time if a value drifts from the contract (typo guard).
_bad = {v for v in PRIORITY_CLASS_MAPPING.values() if v not in PRIORITY_LABELS}
if _bad:
    raise ValueError(
        f"PRIORITY_CLASS_MAPPING has targets not in PRIORITY_LABELS {PRIORITY_LABELS}: {_bad}"
    )


def map_to_priority(source_label: str) -> str:
    """Map one source-dataset ``category`` to a PriorMail priority class.

    Raises:
        KeyError: if ``source_label`` is not a known source category. Raising
            (rather than defaulting) is deliberate: an unseen category means the
            dataset changed and the mapping must be revisited, not silently
            bucketed.
    """
    try:
        return PRIORITY_CLASS_MAPPING[source_label]
    except KeyError:
        raise KeyError(
            f"Unmapped source category {source_label!r}. Known: "
            f"{sorted(PRIORITY_CLASS_MAPPING)}. Update PRIORITY_CLASS_MAPPING in "
            "src/data/loaders.py if the dataset's categories changed."
        ) from None


# Default (v1.0): the original English dataset. The Indonesian translation
# (chairulridjal/...-indonesian) shares the exact schema, so it's a drop-in via
# the ``dataset`` arg. Both are loose-CSV repos (no datasets config).
HF_PRIORITY_DATASET = "jason23322/high-accuracy-email-classifier"
HF_PRIORITY_DATASET_ID = "chairulridjal/high-accuracy-email-classifier-indonesian"
HF_PRIORITY_DATA_FILES = {"train": "train.csv", "test": "test.csv"}
SOURCE_LABEL_COLUMN = "category"


def load_priority_dataset(include_internal: bool = True, dataset: str = HF_PRIORITY_DATASET):
    """Load a public priority-classifier dataset with mapped 4-class labels.

    Adds a ``priority`` column (one of PRIORITY_LABELS) and a ``labels`` column
    (its integer id, for the model head) derived from the source ``category``.

    Args:
        include_internal: if True, also fold the internal Indonesian labeled set
            (ML_PIPELINE.md §7) into the ``train`` split when any labeled files
            exist. The ``test`` split is left untouched so the held-out public
            benchmark stays comparable across runs (§2). No-ops cleanly when the
            internal set is empty.
        dataset: HF dataset id. Defaults to the English original; pass
            ``HF_PRIORITY_DATASET_ID`` for the Indonesian translation. Both share
            the same ``category`` schema, so the class mapping applies unchanged.

    Returns:
        A HuggingFace ``DatasetDict`` with ``train`` and ``test`` splits.
    """
    from datasets import concatenate_datasets, load_dataset

    from src.data.labeled import load_labeled_dataset
    from src.utils.constants import PRIORITY_LABEL2ID

    ds = load_dataset(dataset, data_files=HF_PRIORITY_DATA_FILES)

    def _add_labels(row: dict) -> dict:
        priority = map_to_priority(row[SOURCE_LABEL_COLUMN])
        return {"priority": priority, "labels": PRIORITY_LABEL2ID[priority]}

    ds = ds.map(_add_labels)
    logger.info("Loaded %s with mapped 4-class labels: %s", HF_PRIORITY_DATASET, ds)

    if include_internal:
        internal = load_labeled_dataset()
        if internal is not None:
            # Align columns: keep only what both share + the pipeline needs.
            common = ["subject", "body", "priority", "labels"]
            pub_train = ds["train"].select_columns(common)
            internal = internal.select_columns(common)
            ds["train"] = concatenate_datasets([pub_train, internal])
            logger.info("Merged %d internal records into train (now %d).", internal.num_rows, ds["train"].num_rows)

    return ds


def load_phishing_dataset(download: bool = False):
    """Load the phishing dataset (``ealvaradob/phishing-dataset`` + negatives)."""
    raise NotImplementedError("load_phishing_dataset: not yet implemented.")


def _main() -> None:
    """Smoke test: load the priority dataset and report shape. For the full
    prepare pipeline use ``make data`` (src/data/prepare.py)."""
    argparse.ArgumentParser(description="Smoke-test the priority loader").parse_args()
    logging.basicConfig(level=logging.INFO)
    ds = load_priority_dataset()
    logger.info("Loaded splits: %s", {k: ds[k].num_rows for k in ds})


if __name__ == "__main__":
    _main()
