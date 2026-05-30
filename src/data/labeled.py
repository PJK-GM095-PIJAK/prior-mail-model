"""Loader + schema validation for the internal labeled set (ML_PIPELINE.md §7).

The internal Indonesian work-email set is the domain-adaptation signal that the
public English dataset can't provide. It lives as JSONL in ``data/labeled/``
(gitignored — privacy, §7), one record per line:

    {"id": "...", "subject": "...", "body": "...", "label": "urgent",
     "annotator": "insan", "labeled_at": "2026-05-30T10:00:00"}

This module validates every record against that schema and rejects anything
malformed *loudly* — an unmapped label or a missing field must never slip
silently into training. The actual annotation/agreement tooling (2 annotators,
Cohen's kappa) is built separately; this is the trainable-data boundary.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.utils.constants import PRIORITY_LABEL2ID, PRIORITY_LABELS

logger = logging.getLogger(__name__)

LABELED_DIR = Path("data/labeled")
# Fields every labeled record must carry (§7 schema).
REQUIRED_FIELDS: tuple[str, ...] = ("id", "subject", "body", "label", "annotator", "labeled_at")


class LabelSchemaError(ValueError):
    """Raised when a labeled record violates the §7 schema."""


def validate_record(record: dict, *, source: str = "<record>", line: int | None = None) -> dict:
    """Validate one labeled record against the §7 schema; return it unchanged.

    Raises ``LabelSchemaError`` with a precise location on any violation.
    """
    where = f"{source}" + (f":{line}" if line is not None else "")

    if not isinstance(record, dict):
        raise LabelSchemaError(f"{where}: record is not a JSON object")

    missing = [f for f in REQUIRED_FIELDS if f not in record]
    if missing:
        raise LabelSchemaError(f"{where}: missing required field(s): {missing}")

    label = record["label"]
    if label not in PRIORITY_LABEL2ID:
        raise LabelSchemaError(
            f"{where}: label {label!r} is not a valid priority {PRIORITY_LABELS}"
        )

    # subject may be empty (blank-subject emails are valid), but body must exist.
    if not str(record["body"]).strip():
        raise LabelSchemaError(f"{where}: empty body")
    if not str(record["id"]).strip():
        raise LabelSchemaError(f"{where}: empty id")

    return record


def read_jsonl(path: Path) -> list[dict]:
    """Read + validate one JSONL file. Skips blank lines; raises on bad records."""
    records: list[dict] = []
    for i, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            raise LabelSchemaError(f"{path}:{i}: invalid JSON ({e})") from None
        records.append(validate_record(obj, source=str(path), line=i))
    return records


def load_labeled_dataset(labeled_dir: Path = LABELED_DIR):
    """Load + validate all ``*.jsonl`` in ``labeled_dir`` into a HF ``Dataset``.

    Adds a ``labels`` integer-id column (the model head target) and a ``priority``
    string column, mirroring the public loader so the two can be concatenated.
    Returns ``None`` if no labeled files exist yet (so callers can no-op cleanly).

    Raises:
        LabelSchemaError: if any record is malformed (fail loud, never silent).
    """
    from datasets import Dataset

    files = sorted(labeled_dir.glob("*.jsonl"))
    if not files:
        logger.info("No labeled files in %s yet — internal set is empty.", labeled_dir)
        return None

    records: list[dict] = []
    for f in files:
        batch = read_jsonl(f)
        logger.info("Loaded %d labeled records from %s", len(batch), f.name)
        records.extend(batch)

    if not records:
        return None

    # Detect duplicate ids across batches early (a common labeling slip).
    ids = [r["id"] for r in records]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise LabelSchemaError(f"duplicate record id(s) across labeled files: {sorted(dupes)}")

    for r in records:
        r["priority"] = r["label"]
        r["labels"] = PRIORITY_LABEL2ID[r["label"]]

    ds = Dataset.from_list(records)
    logger.info("Internal labeled set: %d records across %d file(s)", len(records), len(files))
    return ds
