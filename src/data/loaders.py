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
from pathlib import Path

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
HF_PHISHING_EMAIL_SUBSET: str = "texts"  # confirmed: available configs are texts/urls/webs/combined_full/combined_reduced

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


# Column-name candidates for the Nazario CSV (schema-flexible — different mirrors
# name these differently; Naser's Kaggle cut uses sender/subject/body/label).
_NAZ_BODY_COLS = ("body", "text", "text_combined", "email", "message", "content")
_NAZ_SUBJECT_COLS = ("subject",)
_NAZ_SENDER_COLS = ("sender", "from", "sender_email")
_NAZ_LABEL_COLS = ("label", "class", "type")
# Label values that mean "phishing" when a label column is present.
_NAZ_PHISH_LABELS = {"1", "phishing", "phish", "phishing email", "1.0"}


def _read_nazario_phishing(
    csv_path: str | Path, *, sample_size: int | None = None, seed: int = 42
) -> list[dict]:
    """Read real phishing emails from the Nazario CSV into the loader's schema.

    Schema-flexible: locates the body/subject/sender/label columns case-
    insensitively (Naser's Kaggle cut is ``sender,receiver,date,subject,body,
    urls,label``; other mirrors collapse everything into one text column). If a
    label column is present, only phishing rows are kept (the Nazario corpus is
    all phishing, but combined mirrors may not be). When a row carries no separate
    subject/sender but the body looks like a raw RFC 822 message, headers are
    parsed out via ``_parse_raw_email``.

    Returns a list of dicts with keys ``sender_email, subject, body, phishing,
    labels`` (labels = 1). Rows with an empty body are dropped.
    """
    import random as _random

    import pandas as pd

    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, low_memory=False)
    cols = {c.lower().strip(): c for c in df.columns}

    body_c = next((cols[k] for k in _NAZ_BODY_COLS if k in cols), None)
    if body_c is None:
        raise ValueError(
            f"Nazario CSV {csv_path} has no recognizable body/text column "
            f"(looked for {_NAZ_BODY_COLS}); got columns {list(df.columns)}."
        )
    subject_c = next((cols[k] for k in _NAZ_SUBJECT_COLS if k in cols), None)
    sender_c = next((cols[k] for k in _NAZ_SENDER_COLS if k in cols), None)
    label_c = next((cols[k] for k in _NAZ_LABEL_COLS if k in cols), None)

    records: list[dict] = []
    for _, row in df.iterrows():
        if label_c is not None and str(row[label_c]).strip().lower() not in _NAZ_PHISH_LABELS:
            continue
        subject = (row[subject_c] if subject_c else "").strip()
        sender = (row[sender_c] if sender_c else "").strip()
        raw = (row[body_c] or "").strip()
        # If no split headers but the body is a raw email, recover From/Subject.
        if not subject and not sender and raw[:400].lower().lstrip().startswith(
            ("from:", "received:", "subject:", "return-path:", "delivered-to:")
        ):
            p_sender, p_subject, p_body = _parse_raw_email(raw)
            sender, subject, body = p_sender, p_subject, (p_body or raw)
        else:
            body = raw
        if not body.strip():
            continue
        records.append({
            "sender_email": sender, "subject": subject, "body": body,
            "phishing": "phishing", "labels": 1,
        })

    if sample_size is not None and len(records) > sample_size:
        rng = _random.Random(seed)
        idx = sorted(rng.sample(range(len(records)), sample_size))
        records = [records[i] for i in idx]
    return records


def load_phishing_dataset(
    enron_csv_path: str | Path = "emails.csv",
    legit_sample_size: int = ENRON_LEGIT_SAMPLE,
    seed: int = 42,
    hf_dataset: str = HF_PHISHING_DATASET,
    hf_subset: str | None = HF_PHISHING_EMAIL_SUBSET,
    priority_legit_config: str | None = HF_PRIORITY_CONFIG,
    legit_ratio: float = 1.2,
    use_enron: bool = False,
    augmentation_size: int = 0,
    nazario_csv_path: str | Path | None = None,
    use_nazario: bool = False,
    nazario_sample_size: int | None = None,
):
    """Load the phishing training pool: ealvaradob phishing + a diversified legit class.

    Decision log:
    - v1.0 (2026-06-09): legit = Enron only; base = bert-base-multilingual-cased.
      Passed test gates but over-flagged real .eml — legit was Enron-dominated
      (model learned "non-Enron-style => phishing") and the FROM/SUBJECT header
      asymmetry leaked structure.
    - v2 (2026-06-18): legit = ealvaradob benign rows (``texts`` subset is ~62%
      benign) + emails from the team priority dataset
      (``insanar/prior-mail-priority``). The priority set IS the product's real
      legit distribution (transactional, updates, banking-themed legit mail), so
      it directly attacks real-world false positives. The pooled legit class is
      downsampled to ``legit_ratio`` × phishing so the classes stay ~balanced
      (lets us keep modest class weights — see configs/phishing_v2.yaml). Enron
      is OFF by default (``use_enron``); base model is now distilbert-base-uncased.
      Note: ``build_phishing_input`` is body-only in v2, so sender/subject here
      are kept for the schema but not fed to the model.

    Columns in the returned DatasetDict["train"]:
        sender_email  str  — From field (empty string if not in source data)
        subject       str  — Subject field (empty string if not in source data)
        body          str  — Plain-text body
        phishing      str  — "legit" | "phishing"
        labels        int  — 0 = legit, 1 = phishing  (PHISHING_LABEL2ID)

    ``prepare_phishing()`` in ``src/data/prepare.py`` calls this function, then
    applies ``build_phishing_input()`` and ``stratified_split()``.

    Args:
        enron_csv_path: path to the Enron emails CSV (only used if ``use_enron``).
        legit_sample_size: cap on Enron rows sampled (only used if ``use_enron``).
        seed: random seed for sampling/shuffling. Must match the training config.
        hf_dataset: HuggingFace dataset id for the phishing source.
        hf_subset: dataset config/subset name (default ``texts``).
        priority_legit_config: HF config of ``insanar/prior-mail-priority`` to
            add as legit negatives. ``None`` to skip.
        legit_ratio: target legit:phishing ratio. The legit pool is downsampled
            to ``round(legit_ratio * n_phishing)`` if it exceeds that.
        use_enron: include Enron corporate emails in the legit pool (off by
            default per the v2 decision).
        augmentation_size: v2.1 fix B1. If > 0, append this many synthetic
            phishing AND this many synthetic legit rows (balanced, header-
            complete) from ``augment.generate_phishing_augmentation`` — covering
            the tactics the real ``.eml`` acceptance set exposed but the corpora
            miss (BEC, fake-invoice, O365/PayPal credential, …). 0 disables it.
        nazario_csv_path: path to the Nazario phishing CSV (e.g. ``Nazario.csv``
            from the Kaggle ``phishing-email-dataset``). Only used if
            ``use_nazario``.
        use_nazario: v2.2 — add the **real** Jose Nazario phishing corpus to the
            phishing class. Synthetic augmentation hit a ceiling (v2.1 missed
            phishing mimicking routine SaaS/workflow notifications because it only
            saw templated tactics); real, header-bearing phishing adds the
            diversity templates can't. Off by default.
        nazario_sample_size: optional cap on Nazario rows (random, seeded). None
            keeps all.

    Returns:
        A HuggingFace ``DatasetDict`` with a single ``"train"`` split.
        Downstream ``prepare_phishing()`` handles the stratified split.
    """
    import random
    from pathlib import Path as _Path

    import pandas as pd
    from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

    from src.utils.constants import PHISHING_LABEL2ID

    LEGIT = PHISHING_LABEL2ID["legit"]

    # ------------------------------------------------------------------
    # 1. Load phishing source; split into phishing (positive) and benign legit.
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
    phishing_ds = hf_ds.filter(lambda r: r["labels"] != LEGIT)
    n_phishing = phishing_ds.num_rows
    legit_parts: list[Dataset] = [hf_ds.filter(lambda r: r["labels"] == LEGIT)]
    logger.info("HF dataset: %d phishing / %d benign legit", n_phishing, legit_parts[0].num_rows)

    # ------------------------------------------------------------------
    # 2a. Priority-dataset emails as legit negatives (product distribution).
    # ------------------------------------------------------------------
    if priority_legit_config:
        pri = load_dataset(HF_PRIORITY_DATASET, priority_legit_config)
        pri_all = concatenate_datasets([pri[s] for s in pri])

        def _pri_to_legit(row: dict) -> dict:
            return {
                "sender_email": "",
                "subject": row.get("subject") or "",
                "body": row.get("body") or "",
                "phishing": "legit",
                "labels": LEGIT,
            }

        pri_legit = pri_all.map(_pri_to_legit, remove_columns=pri_all.column_names)
        legit_parts.append(pri_legit)
        logger.info("Added %d legit emails from %s:%s", pri_legit.num_rows,
                    HF_PRIORITY_DATASET, priority_legit_config)

    # ------------------------------------------------------------------
    # 2b. Enron legit (off by default — see v2 decision log).
    # ------------------------------------------------------------------
    enron_path = _Path(enron_csv_path) if enron_csv_path else None
    if use_enron and enron_path is not None and enron_path.exists():
        enron_df = pd.read_csv(enron_path, usecols=["message"])
        enron_df = enron_df.dropna(subset=["message"]).reset_index(drop=True)
        sample_size = min(legit_sample_size, len(enron_df))
        rng = random.Random(seed)
        sample_idx = sorted(rng.sample(range(len(enron_df)), sample_size))
        enron_df = enron_df.iloc[sample_idx].reset_index(drop=True)
        enron_records = []
        for msg_str in enron_df["message"]:
            sender, subject, body = _parse_raw_email(str(msg_str))
            enron_records.append({
                "sender_email": sender, "subject": subject, "body": body,
                "phishing": "legit", "labels": LEGIT,
            })
        legit_parts.append(Dataset.from_list(enron_records))
        logger.info("Added %d Enron legit records", len(enron_records))

    # ------------------------------------------------------------------
    # 3. Pool + balance legit to ~legit_ratio × phishing (keeps weights modest).
    # ------------------------------------------------------------------
    # Sources disagree on string subtype (ealvaradob -> string, priority ->
    # large_string); cast everyone to one schema so concatenation aligns.
    from datasets import Features, Value

    schema = Features({
        "sender_email": Value("string"),
        "subject": Value("string"),
        "body": Value("string"),
        "phishing": Value("string"),
        "labels": Value("int64"),
    })
    phishing_ds = phishing_ds.cast(schema)
    legit_parts = [p.cast(schema) for p in legit_parts]

    # ------------------------------------------------------------------
    # 2c. Synthetic augmentation (v2.1 fix B1) — header-complete, both classes.
    # Added AFTER the schema cast so it aligns; n_phishing is recomputed so the
    # legit-balancing target below accounts for the extra phishing rows.
    # ------------------------------------------------------------------
    if augmentation_size > 0:
        from src.data.augment import generate_phishing_augmentation

        aug = generate_phishing_augmentation(n_per_class=augmentation_size, seed=seed).cast(schema)
        aug_phishing = aug.filter(lambda r: r["labels"] != LEGIT)
        aug_legit = aug.filter(lambda r: r["labels"] == LEGIT)
        phishing_ds = concatenate_datasets([phishing_ds, aug_phishing])
        legit_parts.append(aug_legit)
        n_phishing = phishing_ds.num_rows
        logger.info(
            "Added synthetic augmentation: +%d phishing / +%d legit (n_phishing now %d)",
            aug_phishing.num_rows, aug_legit.num_rows, n_phishing,
        )

    # ------------------------------------------------------------------
    # 2d. Nazario real phishing corpus (v2.2). Real, header-bearing phishing
    # emails to break the synthetic-augmentation ceiling — added to the PHISHING
    # side only; n_phishing is recomputed so legit balancing accounts for them.
    # ------------------------------------------------------------------
    naz_path = _Path(nazario_csv_path) if nazario_csv_path else None
    if use_nazario and naz_path is not None and naz_path.exists():
        naz_records = _read_nazario_phishing(naz_path, sample_size=nazario_sample_size, seed=seed)
        if naz_records:
            naz_ds = Dataset.from_list(naz_records).cast(schema)
            phishing_ds = concatenate_datasets([phishing_ds, naz_ds])
            n_phishing = phishing_ds.num_rows
            logger.info(
                "Added %d Nazario real phishing emails (n_phishing now %d)",
                naz_ds.num_rows, n_phishing,
            )
        else:
            logger.warning("Nazario CSV %s yielded 0 usable rows — skipping.", naz_path)
    elif use_nazario:
        logger.warning(
            "use_nazario=True but Nazario CSV not found at %s — skipping "
            "(attach the Kaggle phishing-email-dataset and copy Nazario.csv).",
            naz_path,
        )

    legit_pool = concatenate_datasets(legit_parts) if len(legit_parts) > 1 else legit_parts[0]
    target_legit = round(legit_ratio * n_phishing)
    if legit_pool.num_rows > target_legit:
        legit_pool = legit_pool.shuffle(seed=seed).select(range(target_legit))
        logger.info("Downsampled legit pool to %d (target ratio %.2f× phishing)",
                    target_legit, legit_ratio)

    combined = concatenate_datasets([phishing_ds, legit_pool])

    # ------------------------------------------------------------------
    # 4. Deduplicate on body fingerprint to prevent train/val/test leakage.
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
