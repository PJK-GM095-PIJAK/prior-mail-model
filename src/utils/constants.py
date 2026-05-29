"""Shared constants for PriorMail ML.

The priority labels here are the **contract** with the backend: they must match
the ``priority_level`` Postgres enum in ``docs/DATA_MODELS.md`` exactly. Do not
reorder or rename without a coordinated change across all repos.
"""

from __future__ import annotations

# Priority classes — order is the label-index mapping used by the model head.
# MUST equal docs/DATA_MODELS.md `priority_level`: urgent | high | normal | low
PRIORITY_LABELS: tuple[str, ...] = ("urgent", "high", "normal", "low")

# label string -> integer id (model output index)
PRIORITY_LABEL2ID: dict[str, int] = {lbl: i for i, lbl in enumerate(PRIORITY_LABELS)}
# integer id -> label string
PRIORITY_ID2LABEL: dict[int, str] = {i: lbl for lbl, i in PRIORITY_LABEL2ID.items()}

# Phishing is binary: index 1 is the positive (phishing) class.
PHISHING_LABELS: tuple[str, ...] = ("legit", "phishing")
PHISHING_LABEL2ID: dict[str, int] = {lbl: i for i, lbl in enumerate(PHISHING_LABELS)}
PHISHING_ID2LABEL: dict[int, str] = {i: lbl for lbl, i in PHISHING_LABEL2ID.items()}

# Base model (LOCKED — CLAUDE.md §4 / ML_PIPELINE.md §2).
BASE_MODEL_NAME: str = "indobenchmark/indobert-base-p1"

# Tokenization (ML_PIPELINE.md §2).
MAX_SEQ_LENGTH: int = 512

# Placeholder tokens injected during preprocessing (ML_PIPELINE.md §2).
URL_TOKEN: str = "[URL]"
EMAIL_TOKEN: str = "[EMAIL]"
