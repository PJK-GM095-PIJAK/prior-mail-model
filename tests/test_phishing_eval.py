"""Tests for phishing eval gate logic, threshold selection, and metric helpers.

All tests are GPU-free — no checkpoint or model loading. Mirrors the pattern
in tests/test_eval.py (priority classifier). Keep ``make test`` green.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.eval_phishing import (
    PRECISION_GATE,
    RECALL_GATE,
    check_gates,
    select_threshold,
)


# ---------------------------------------------------------------------------
# check_gates — pure logic, no model
# ---------------------------------------------------------------------------

def test_check_gates_all_pass() -> None:
    metrics = {
        "recall_phishing": 0.97,
        "precision_phishing": 0.85,
        "p95_ms": 200.0,
    }
    result = check_gates(metrics)
    assert result["all_passed"] is True
    assert result["gates"] == {"recall": True, "precision": True, "latency_p95": True}


def test_check_gates_recall_fails() -> None:
    metrics = {
        "recall_phishing": 0.89,   # below 0.95 gate
        "precision_phishing": 0.90,
        "p95_ms": 150.0,
    }
    result = check_gates(metrics)
    assert result["all_passed"] is False
    assert result["gates"]["recall"] is False
    assert result["gates"]["precision"] is True
    assert result["gates"]["latency_p95"] is True


def test_check_gates_precision_fails() -> None:
    metrics = {
        "recall_phishing": 0.97,
        "precision_phishing": 0.72,   # below 0.80 gate
        "p95_ms": 150.0,
    }
    result = check_gates(metrics)
    assert result["all_passed"] is False
    assert result["gates"]["precision"] is False
    assert result["gates"]["recall"] is True


def test_check_gates_latency_fails() -> None:
    metrics = {
        "recall_phishing": 0.97,
        "precision_phishing": 0.85,
        "p95_ms": 600.0,   # over 500ms gate
    }
    result = check_gates(metrics)
    assert result["all_passed"] is False
    assert result["gates"]["latency_p95"] is False


def test_check_gates_all_fail() -> None:
    metrics = {
        "recall_phishing": 0.80,
        "precision_phishing": 0.70,
        "p95_ms": 750.0,
    }
    result = check_gates(metrics)
    assert result["all_passed"] is False
    assert not any(result["gates"].values())


def test_check_gates_boundary_values_pass() -> None:
    """Exact boundary values must pass (gate is >=, not >)."""
    metrics = {
        "recall_phishing": RECALL_GATE,        # exactly 0.95
        "precision_phishing": PRECISION_GATE,  # exactly 0.80
        "p95_ms": 499.9,
    }
    result = check_gates(metrics)
    assert result["all_passed"] is True


# ---------------------------------------------------------------------------
# select_threshold — pure logic, no model
# ---------------------------------------------------------------------------

def test_select_threshold_perfect_separation() -> None:
    """With perfectly separable scores, threshold finds a clean split."""
    # 100 phishing at score 0.9, 100 legit at score 0.1.
    scores = [0.9] * 100 + [0.1] * 100
    labels = [1] * 100 + [0] * 100
    t = select_threshold(scores, labels)
    # Threshold should be between the two score clusters.
    assert 0.1 < t < 0.95
    # Applying it: should achieve recall >= 0.95.
    preds = [1 if s >= t else 0 for s in scores]
    n_phishing = sum(1 for l in labels if l == 1)
    tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
    assert tp / n_phishing >= RECALL_GATE


def test_select_threshold_maximises_precision() -> None:
    """Among thresholds meeting recall >= 0.95, selects the one with best precision."""
    rng = np.random.default_rng(42)
    phishing_scores = rng.beta(8, 2, 300).tolist()   # concentrated near 1
    legit_scores = rng.beta(2, 6, 300).tolist()       # concentrated near 0
    scores = phishing_scores + legit_scores
    labels = [1] * 300 + [0] * 300

    t = select_threshold(scores, labels)
    preds = [1 if s >= t else 0 for s in scores]

    from sklearn.metrics import precision_score, recall_score
    recall = recall_score(labels, preds, pos_label=1, zero_division=0)
    precision = precision_score(labels, preds, pos_label=1, zero_division=0)
    assert recall >= RECALL_GATE
    assert precision >= PRECISION_GATE


def test_select_threshold_fallback_when_impossible() -> None:
    """If recall >= 0.95 is unachievable, falls back to 0.5 without raising."""
    # All scores are identical — can't separate classes.
    scores = [0.5] * 200
    labels = [1] * 100 + [0] * 100
    t = select_threshold(scores, labels)
    # Must return a valid threshold, not raise.
    assert 0.0 <= t <= 1.0


def test_select_threshold_all_phishing_achievable() -> None:
    """When all examples are phishing, any threshold achieves recall = 1.0."""
    scores = [0.8] * 50 + [0.3] * 50
    labels = [1] * 100  # all phishing
    t = select_threshold(scores, labels)
    preds = [1 if s >= t else 0 for s in scores]
    from sklearn.metrics import recall_score
    assert recall_score(labels, preds, pos_label=1, zero_division=0) >= RECALL_GATE


def test_select_threshold_returns_float() -> None:
    """Return type is always a Python float."""
    scores = [float(x) for x in range(100)]
    labels = [1 if i % 2 == 0 else 0 for i in range(100)]
    t = select_threshold(scores, labels)
    assert isinstance(t, float)


def test_select_threshold_in_valid_range() -> None:
    """Returned threshold must be in [0, 1]."""
    rng = np.random.default_rng(7)
    scores = rng.random(200).tolist()
    labels = rng.integers(0, 2, 200).tolist()
    t = select_threshold(scores, labels)
    assert 0.0 <= t <= 1.0
