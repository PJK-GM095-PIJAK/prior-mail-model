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


# ---------------------------------------------------------------------------
# Phishing dataset constants
# ---------------------------------------------------------------------------

# Primary phishing training data (HuggingFace).
HF_PHISHING_DATASET: str = "ealvaradob/phishing-dataset"

# Email-specific subset of the above dataset (~18 K email samples).
# The full combined dataset includes URLs/SMS/websites — use this subset for
# an email phishing model.  Verify the exact name in
# notebooks/02_phishing_overview.ipynb and update here if it differs.
HF_PHISHING_EMAIL_SUBSET: str = "emails"

# How many Enron legit emails to sample as the negative class.
# With ~18 K phishing emails in the email subset, 20 K legit gives ~1:1
# balance before class weights push recall further.
ENRON_LEGIT_SAMPLE: int = 20_000


def _parse_raw_email(text: str) -> tuple[str, str, str]:
    """Extract (sender_email, subject, body) from a raw email string.

    Handles two cases:
    - RFC 2822 formatted string (Enron emails, some phishing corpora): parse
      headers to get From / Subject / body.
    - Plain body text (no detectable headers): return ("", "", text).
    """
    import email as _email_lib

    try:
        msg = _email_lib.message_from_string(text)
        sender = (msg.get("From") or "").strip()
        subject = (msg.get("Subject") or "").strip()

        # Only treat as a parsed RFC 2822 email if at least one header found.
        if not sender and not subject:
            return "", "", text

        # Extract the plain-text body.
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        body = payload.decode("utf-8", errors="ignore")
                    break
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                body = payload.decode("utf-8", errors="ignore")
            else:
                body = msg.get_payload() or ""
        return sender, subject, body
    except Exception:
        # Malformed email string — treat entire text as body.
        return "", "", text


def load_phishing_dataset(
    enron_csv_path: str | Path = "emails.csv",
    legit_sample_size: int = ENRON_LEGIT_SAMPLE,
    seed: int = 42,
    hf_dataset: str = HF_PHISHING_DATASET,
    hf_subset: str | None = HF_PHISHING_EMAIL_SUBSET,
):
    """Load the phishing dataset (``ealvaradob/phishing-dataset``) + Enron legit emails.

    Decision log (feat/phishing-model PR, 2026-06-09 — §2 of PHISHING_MODEL_GUIDE):
    - Negative (legit) class source: Enron corporate email corpus (``emails.csv``).
    - Imbalance handling: weighted cross-entropy (class weights in training config).
    - Base model: bert-base-multilingual-cased (English-capable; §11 decision,
      approved by Insan).

    Columns in the returned DatasetDict["train"]:
        sender_email  str  — From field (empty string if not in source data)
        subject       str  — Subject field (empty string if not in source data)
        body          str  — Plain-text body
        phishing      str  — "legit" | "phishing"
        labels        int  — 0 = legit, 1 = phishing  (PHISHING_LABEL2ID)

    Preprocessing applied here:
    - RFC 2822 email strings (both HF dataset and Enron) are parsed via
      ``_parse_raw_email`` to extract sender / subject / body fields.
    - Deduplication on the first 200 chars of body to remove near-duplicates
      before splitting (prevents leakage across train/val/test).

    ``prepare_phishing()`` in ``src/data/prepare.py`` calls this function, then
    applies ``build_phishing_input()`` and ``stratified_split()``.

    Args:
        enron_csv_path: path to the Enron emails CSV (``file``, ``message`` cols).
            Pass ``None`` or a non-existent path to skip Enron (legit class will
            come from HF benign rows only — expect heavier imbalance).
        legit_sample_size: how many Enron rows to sample.  Sampling is random
            with ``seed`` for reproducibility.
        seed: random seed for Enron sampling.  Must match the training config.
        hf_dataset: HuggingFace dataset id for the phishing source.
        hf_subset: dataset config/subset name.  Set to ``None`` to load the
            default config (may mix email / URL / SMS records — consider
            filtering by dataset type in that case).

    Returns:
        A HuggingFace ``DatasetDict`` with a single ``"train"`` split.
        Downstream ``prepare_phishing()`` handles the stratified train/val/test
        split.
    """
    import random
    from pathlib import Path as _Path

    import pandas as pd
    from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

    from src.utils.constants import PHISHING_LABEL2ID

    # ------------------------------------------------------------------
    # 1. Load phishing examples from HuggingFace
    # ------------------------------------------------------------------
    try:
        hf_kwargs: dict = {"trust_remote_code": True}
        if hf_subset:
            hf_kwargs["name"] = hf_subset
        raw = load_dataset(hf_dataset, **hf_kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load HF dataset {hf_dataset!r} (subset={hf_subset!r}). "
            "Check the dataset page for available config names and update "
            "HF_PHISHING_EMAIL_SUBSET in src/data/loaders.py. "
            f"Original error: {exc}"
        ) from exc

    # ealvaradob/phishing-dataset ships only a "train" split — pool all available.
    hf_rows = concatenate_datasets([raw[s] for s in raw])
    logger.info("Loaded %d rows from %s (subset=%s)", hf_rows.num_rows, hf_dataset, hf_subset)

    if "text" not in hf_rows.column_names or "label" not in hf_rows.column_names:
        raise ValueError(
            f"Expected columns 'text' and 'label' in {hf_dataset}; "
            f"got {hf_rows.column_names}. "
            "Inspect in notebooks/02_phishing_overview.ipynb and update the loader."
        )

    def _hf_row_to_fields(row: dict) -> dict:
        sender, subject, body = _parse_raw_email(row["text"])
        label_int = int(row["label"])
        phishing_str = "phishing" if label_int == 1 else "legit"
        return {
            "sender_email": sender,
            "subject": subject,
            "body": body,
            "phishing": phishing_str,
            "labels": PHISHING_LABEL2ID[phishing_str],
        }

    hf_ds = hf_rows.map(_hf_row_to_fields, remove_columns=hf_rows.column_names)
    n_phishing_hf = sum(1 for x in hf_ds["labels"] if x == 1)
    n_legit_hf = sum(1 for x in hf_ds["labels"] if x == 0)
    logger.info("HF dataset: %d phishing / %d legit", n_phishing_hf, n_legit_hf)

    # ------------------------------------------------------------------
    # 2. Load Enron legit emails (negative class supplement)
    # ------------------------------------------------------------------
    enron_path = _Path(enron_csv_path) if enron_csv_path else None
    enron_ds: Dataset | None = None

    if enron_path is not None and enron_path.exists():
        enron_df = pd.read_csv(enron_path, usecols=["message"])
        enron_df = enron_df.dropna(subset=["message"]).reset_index(drop=True)
        # Sample before parsing to avoid processing all 517 K rows.
        sample_size = min(legit_sample_size, len(enron_df))
        rng = random.Random(seed)
        sample_idx = sorted(rng.sample(range(len(enron_df)), sample_size))
        enron_df = enron_df.iloc[sample_idx].reset_index(drop=True)
        logger.info("Sampled %d rows from Enron CSV (%s)", len(enron_df), enron_path)

        enron_records = []
        for msg_str in enron_df["message"]:
            sender, subject, body = _parse_raw_email(str(msg_str))
            enron_records.append({
                "sender_email": sender,
                "subject": subject,
                "body": body,
                "phishing": "legit",
                "labels": PHISHING_LABEL2ID["legit"],
            })
        enron_ds = Dataset.from_list(enron_records)
        logger.info("Built %d Enron legit records", enron_ds.num_rows)
    else:
        logger.warning(
            "Enron CSV not found at %s — legit class from HF benign rows only. "
            "Expect heavier class imbalance; consider increasing phishing_class_multiplier.",
            enron_path,
        )

    # ------------------------------------------------------------------
    # 3. Combine
    # ------------------------------------------------------------------
    combined = concatenate_datasets([hf_ds, enron_ds]) if enron_ds is not None else hf_ds

    # ------------------------------------------------------------------
    # 4. Deduplicate on body fingerprint to prevent train/val/test leakage.
    #    See audit note: phishing corpora often contain near-duplicate emails.
    # ------------------------------------------------------------------
    before = combined.num_rows
    seen: set[str] = set()
    keep: list[int] = []
    for i, body in enumerate(combined["body"]):
        key = body.strip()[:200]  # fingerprint on first 200 chars
        if key not in seen:
            seen.add(key)
            keep.append(i)
    combined = combined.select(keep)
    n_phishing_final = sum(1 for x in combined["labels"] if x == 1)
    n_legit_final = sum(1 for x in combined["labels"] if x == 0)
    logger.info(
        "After dedup: %d rows (removed %d duplicates) — %d phishing / %d legit",
        combined.num_rows,
        before - combined.num_rows,
        n_phishing_final,
        n_legit_final,
    )

    return DatasetDict({"train": combined})


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
