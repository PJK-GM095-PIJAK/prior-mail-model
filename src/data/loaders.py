"""Dataset loaders for the priority and phishing models.

The priority dataset (``insanar/prior-mail-priority``) is the team-curated,
canonical source: it already carries a direct ``label`` in our 4-class scheme
(``urgent | high | normal | low``) and ships its own ``train/validation/test``
splits. There is no external-category -> priority mapping anymore — the public
topical dataset and the synthetic bootstrap were folded into this dataset
upstream (see its ``label_source`` column: ``hf_dataset`` | ``synthetic``).
"""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)

# Canonical priority dataset (CLAUDE.md §7). Already priority-labelled + split.
HF_PRIORITY_DATASET = "insanar/prior-mail-priority"
# HF config (dataset "version"). ``v2`` is the current curated cut; ``default``/
# ``v1`` are the larger earlier build. Pinned here so runs are reproducible.
HF_PRIORITY_CONFIG = "v4"
# Column on the source dataset holding the priority string (our 4-class label).
SOURCE_LABEL_COLUMN = "label"


def load_priority_dataset(config: str = HF_PRIORITY_CONFIG):
    """Load the priority dataset with model-ready label columns.

    Reads the dataset's direct ``label`` (one of PRIORITY_LABELS) and adds a
    ``priority`` column (the same string) plus a ``labels`` column (its integer
    id, for the model head). The dataset's own ``train/validation/test`` splits
    are returned unchanged — we do not re-split (the splits are published and
    versioned, and the set already contains synthetic rows that must not leak
    into the held-out test split).

    Args:
        config: HF dataset config to load (default ``v2``).

    Returns:
        A HuggingFace ``DatasetDict`` with ``train`` / ``validation`` / ``test``.
    """
    from datasets import load_dataset

    from src.utils.constants import PRIORITY_LABEL2ID, PRIORITY_LABELS

    ds = load_dataset(HF_PRIORITY_DATASET, config)

    def _add_labels(row: dict) -> dict:
        priority = row[SOURCE_LABEL_COLUMN]
        if priority not in PRIORITY_LABEL2ID:
            # Fail loud: an unexpected label means the dataset drifted from the
            # backend contract (PRIORITY_LABELS), not something to bucket silently.
            raise ValueError(
                f"Row has label {priority!r}, not a valid priority {PRIORITY_LABELS}. "
                f"Dataset {HF_PRIORITY_DATASET}:{config} drifted from the contract."
            )
        return {"priority": priority, "labels": PRIORITY_LABEL2ID[priority]}

    ds = ds.map(_add_labels)
    logger.info("Loaded %s:%s with 4-class labels: %s", HF_PRIORITY_DATASET, config, ds)
    return ds


def load_phishing_dataset(download: bool = False):
    """Load the phishing dataset (``ealvaradob/phishing-dataset`` + negatives)."""
    raise NotImplementedError("load_phishing_dataset: not yet implemented.")


def _main() -> None:
    """Smoke test: load the priority dataset and report shape. For the full
    prepare pipeline use ``make data`` (src/data/prepare.py)."""
    parser = argparse.ArgumentParser(description="Smoke-test the priority loader")
    parser.add_argument("--config", default=HF_PRIORITY_CONFIG, help="HF dataset config")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    ds = load_priority_dataset(config=args.config)
    logger.info("Loaded splits: %s", {k: ds[k].num_rows for k in ds})


if __name__ == "__main__":
    _main()
