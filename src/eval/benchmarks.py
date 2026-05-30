"""Shared benchmark utilities (latency measurement, metric helpers).

Used by both eval harnesses. The CPU latency benchmark must reflect Render's
instance (ML_PIPELINE.md §2/§3: p95 < 500 ms per email on CPU).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile (q in [0, 100]). Avoids a numpy dep here."""
    if not values:
        raise ValueError("percentile() of empty sequence")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = (q / 100.0) * (len(s) - 1)
    lo = int(rank)
    frac = rank - lo
    if lo + 1 >= len(s):
        return s[lo]
    return s[lo] + frac * (s[lo + 1] - s[lo])


def measure_latency_ms(
    predict_fn: Callable[[str], object],
    inputs: list[str],
    *,
    warmup: int = 3,
    q: float = 95.0,
) -> dict[str, float]:
    """Time ``predict_fn`` on one input at a time and summarize latency in ms.

    The function is intentionally model-agnostic: pass a closure that runs a
    single-email forward pass on CPU. The caller is responsible for putting the
    model on CPU and in eval mode (so this reflects the Render target, §2/§3).

    Returns a dict with ``mean_ms``, ``p50_ms``, ``p95_ms`` (the gate metric is
    p95). The percentile ``q`` is configurable but defaults to 95.
    """
    if not inputs:
        raise ValueError("measure_latency_ms requires at least one input")

    # Warm up (first calls pay one-time costs: lazy init, allocator, caches).
    for x in inputs[:warmup]:
        predict_fn(x)

    timings: list[float] = []
    for x in inputs:
        start = time.perf_counter()
        predict_fn(x)
        timings.append((time.perf_counter() - start) * 1000.0)

    summary = {
        "mean_ms": sum(timings) / len(timings),
        "p50_ms": percentile(timings, 50.0),
        f"p{int(q)}_ms": percentile(timings, q),
        "n": float(len(timings)),
    }
    logger.info("Latency over %d inputs: %s", len(timings), summary)
    return summary
