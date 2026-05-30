"""Tests for inter-annotator agreement tooling (§7)."""

import pytest
from src.data.agreement import (
    cohen_kappa,
    compare_annotators,
)


def _rec(id_, label, subject="s"):
    return {
        "id": id_,
        "subject": subject,
        "body": "b",
        "label": label,
        "annotator": "x",
        "labeled_at": "2026-05-30T10:00:00",
    }


def test_perfect_agreement_kappa_one():
    labels = ["urgent", "high", "normal", "low", "urgent", "high"]
    assert cohen_kappa(labels, labels) == pytest.approx(1.0)


def test_kappa_length_mismatch_raises():
    with pytest.raises(ValueError):
        cohen_kappa(["urgent"], ["urgent", "high"])


def test_kappa_empty_raises():
    with pytest.raises(ValueError):
        cohen_kappa([], [])


def test_compare_aligns_by_shared_id():
    a = [_rec("e1", "urgent"), _rec("e2", "high"), _rec("e3", "low")]
    b = [_rec("e2", "high"), _rec("e3", "normal"), _rec("e4", "low")]  # e4 unique to b
    report = compare_annotators(a, b)
    # shared ids: e2, e3 → 2 scored, e1/e4 ignored
    assert report.n_shared == 2
    assert report.n_agree == 1  # e2 agrees, e3 disagrees


def test_disagreements_listed():
    a = [_rec("e1", "urgent", "Server down"), _rec("e2", "low")]
    b = [_rec("e1", "high", "Server down"), _rec("e2", "low")]
    report = compare_annotators(a, b)
    assert len(report.disagreements) == 1
    d = report.disagreements[0]
    assert d.id == "e1" and d.label_a == "urgent" and d.label_b == "high"
    assert d.subject == "Server down"


def test_percent_and_target_with_varied_labels():
    # Perfect agreement across MULTIPLE classes -> kappa 1.0, meets target.
    labels = ["urgent", "high", "normal", "low"] * 3
    a = [_rec(f"e{i}", lbl) for i, lbl in enumerate(labels)]
    b = [_rec(f"e{i}", lbl) for i, lbl in enumerate(labels)]
    report = compare_annotators(a, b)
    assert report.percent_agree == 1.0
    assert report.meets_target is True
    assert report.kappa == pytest.approx(1.0)


def test_single_class_kappa_is_undefined_despite_full_agreement():
    # Documented quirk: if both annotators use ONLY one class, kappa is
    # UNDEFINED (nan) — agreement beyond chance can't be measured — even at
    # 100% raw agreement. meets_target must be False (don't pass on nan).
    import math

    a = [_rec(f"e{i}", "normal") for i in range(10)]
    b = [_rec(f"e{i}", "normal") for i in range(10)]
    report = compare_annotators(a, b)
    assert report.percent_agree == 1.0
    assert math.isnan(report.kappa)
    assert report.meets_target is False


def test_no_shared_ids_raises():
    a = [_rec("e1", "urgent")]
    b = [_rec("e2", "high")]
    with pytest.raises(ValueError, match="share no ids"):
        compare_annotators(a, b)
