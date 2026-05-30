"""Tests for the eval gate logic, metrics, and latency helpers.

The model-loading path needs a real checkpoint and isn't unit-tested here; the
pure logic (gates, metrics, percentiles) is fully covered.
"""

import pytest
from src.eval.benchmarks import measure_latency_ms, percentile
from src.eval.eval_priority import check_gates, compute_metrics


def test_percentile_basic() -> None:
    assert percentile([10], 95) == 10
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    # p95 of 1..100 lands near the top.
    assert 95 <= percentile(list(range(1, 101)), 95) <= 96


def test_measure_latency_shape() -> None:
    out = measure_latency_ms(lambda x: x, ["a", "b", "c", "d", "e"], warmup=1)
    assert set(out) == {"mean_ms", "p50_ms", "p95_ms", "n"}
    assert out["n"] == 5
    assert out["mean_ms"] >= 0


def test_measure_latency_empty_raises() -> None:
    with pytest.raises(ValueError):
        measure_latency_ms(lambda x: x, [])


def test_compute_metrics_perfect() -> None:
    y = [0, 1, 2, 3, 0, 1, 2, 3]
    m = compute_metrics(y, y)
    assert m["macro_f1"] == 1.0
    assert all(m[f"recall_{n}"] == 1.0 for n in ("urgent", "high", "normal", "low"))
    assert len(m["confusion_matrix"]) == 4


def test_check_gates_all_pass() -> None:
    metrics = {
        "macro_f1": 0.85,
        "recall_urgent": 0.7,
        "recall_high": 0.7,
        "recall_normal": 0.9,
        "recall_low": 0.9,
        "p95_ms": 200.0,
    }
    result = check_gates(metrics)
    assert result["all_passed"] is True
    assert result["gates"] == {"macro_f1": True, "per_class_recall": True, "latency_p95": True}


def test_check_gates_fail_one_class_below_recall() -> None:
    metrics = {
        "macro_f1": 0.85,
        "recall_urgent": 0.50,  # below 0.65 gate
        "recall_high": 0.9,
        "recall_normal": 0.9,
        "recall_low": 0.9,
        "p95_ms": 200.0,
    }
    result = check_gates(metrics)
    assert result["all_passed"] is False
    assert result["gates"]["per_class_recall"] is False
    assert result["per_class_recall_detail"]["urgent"] is False


def test_check_gates_fail_latency() -> None:
    metrics = {
        "macro_f1": 0.95,
        "recall_urgent": 0.9,
        "recall_high": 0.9,
        "recall_normal": 0.9,
        "recall_low": 0.9,
        "p95_ms": 750.0,  # over 500ms gate
    }
    result = check_gates(metrics)
    assert result["all_passed"] is False
    assert result["gates"]["latency_p95"] is False
