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


def test_priority_class_mapping_targets_are_valid() -> None:
    """Every mapped target must be a real priority label (no typos)."""
    from src.data.loaders import PRIORITY_CLASS_MAPPING

    assert PRIORITY_CLASS_MAPPING, "mapping must not be empty"
    assert set(PRIORITY_CLASS_MAPPING.values()) <= set(PRIORITY_LABELS)


def test_priority_class_mapping_covers_known_categories() -> None:
    """Guard the exact source categories the mapping was built against."""
    from src.data.loaders import PRIORITY_CLASS_MAPPING

    expected = {"verify_code", "updates", "forum", "social_media", "promotions", "spam"}
    assert set(PRIORITY_CLASS_MAPPING) == expected


def test_map_to_priority_raises_on_unknown() -> None:
    import pytest
    from src.data.loaders import map_to_priority

    assert map_to_priority("spam") == "low"
    with pytest.raises(KeyError):
        map_to_priority("not_a_real_category")
