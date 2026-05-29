"""Shared benchmark utilities (latency measurement, metric helpers).

Used by both eval harnesses. The CPU latency benchmark must reflect Render's
instance (ML_PIPELINE.md §2/§3: p95 < 500 ms per email on CPU).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def measure_cpu_latency_p95(model, inputs: list[str]) -> float:
    """Return p95 single-email inference latency in milliseconds, on CPU. Stub."""
    raise NotImplementedError("measure_cpu_latency_p95: not yet implemented.")
