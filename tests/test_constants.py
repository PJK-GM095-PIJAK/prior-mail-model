"""Guard the backend contract: priority labels must not drift.

These labels are mirrored in the Postgres `priority_level` enum
(docs/DATA_MODELS.md §2). If this test fails, the model and backend disagree.
"""

from src.utils.constants import (
    PRIORITY_ID2LABEL,
    PRIORITY_LABEL2ID,
    PRIORITY_LABELS,
)


def test_priority_labels_match_contract() -> None:
    # Exact set + order locked by docs/DATA_MODELS.md priority_level enum.
    assert PRIORITY_LABELS == ("urgent", "high", "normal", "low")


def test_priority_label_id_roundtrip() -> None:
    for label, idx in PRIORITY_LABEL2ID.items():
        assert PRIORITY_ID2LABEL[idx] == label
    assert len(PRIORITY_LABEL2ID) == len(PRIORITY_LABELS)
