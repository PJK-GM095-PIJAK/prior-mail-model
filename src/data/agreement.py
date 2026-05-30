"""Inter-annotator agreement tooling for the internal labeled set (§7).

Implements the 2-annotator protocol: each email is labeled independently by two
annotators; we compute Cohen's kappa (target ≥ 0.7), surface disagreements for
the weekly sync, and — once disagreements are resolved — emit the final
single-label JSONL the loader consumes.

Input: two annotator JSONL files in the §7 schema (same ``id`` set, possibly
overlapping). Each annotator's file is itself schema-valid (see
``src/data/labeled.py``); this module aligns them by ``id``.

CLI:
  python -m src.data.agreement --a annotatorA.jsonl --b annotatorB.jsonl
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.data.labeled import read_jsonl
from src.utils.constants import PRIORITY_LABELS

logger = logging.getLogger(__name__)

KAPPA_TARGET = 0.7  # §7 inter-annotator agreement target


@dataclass
class Disagreement:
    """One email where the two annotators assigned different labels."""

    id: str
    label_a: str
    label_b: str
    subject: str


@dataclass
class AgreementReport:
    """Result of comparing two annotators over their shared email ids."""

    n_shared: int
    kappa: float
    n_agree: int
    disagreements: list[Disagreement]

    @property
    def percent_agree(self) -> float:
        return self.n_agree / self.n_shared if self.n_shared else 0.0

    @property
    def meets_target(self) -> bool:
        return self.kappa >= KAPPA_TARGET


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """Cohen's kappa between two equal-length label sequences.

    Uses the fixed PRIORITY_LABELS set so kappa is comparable across batches even
    when a batch happens to omit a class.

    Note: kappa measures agreement *beyond chance*. If both annotators use only
    one class, kappa is 0.0 even at 100% raw agreement (there is no variance to
    measure). Read kappa alongside ``percent_agree``, not in isolation.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("label sequences must be the same length")
    if not labels_a:
        raise ValueError("cannot compute kappa over zero shared items")
    from sklearn.metrics import cohen_kappa_score

    return float(cohen_kappa_score(labels_a, labels_b, labels=list(PRIORITY_LABELS)))


def compare_annotators(records_a: list[dict], records_b: list[dict]) -> AgreementReport:
    """Align two annotators' records by ``id`` and compute the agreement report.

    Only ids present in BOTH annotators' files are scored (an email needs two
    labels to measure agreement). Ids unique to one annotator are ignored here.
    """
    by_id_a = {r["id"]: r for r in records_a}
    by_id_b = {r["id"]: r for r in records_b}
    shared = sorted(set(by_id_a) & set(by_id_b))
    if not shared:
        raise ValueError("annotators share no ids — nothing to compare")

    labels_a = [by_id_a[i]["label"] for i in shared]
    labels_b = [by_id_b[i]["label"] for i in shared]

    disagreements = [
        Disagreement(
            id=i,
            label_a=by_id_a[i]["label"],
            label_b=by_id_b[i]["label"],
            subject=str(by_id_a[i].get("subject", "")),
        )
        for i in shared
        if by_id_a[i]["label"] != by_id_b[i]["label"]
    ]
    n_agree = len(shared) - len(disagreements)

    return AgreementReport(
        n_shared=len(shared),
        kappa=cohen_kappa(labels_a, labels_b),
        n_agree=n_agree,
        disagreements=disagreements,
    )


def _main() -> None:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Compute inter-annotator agreement (§7)")
    parser.add_argument("--a", required=True, type=Path, help="annotator A's JSONL")
    parser.add_argument("--b", required=True, type=Path, help="annotator B's JSONL")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    report = compare_annotators(read_jsonl(args.a), read_jsonl(args.b))

    import math

    kappa_str = "undefined (only one class used)" if math.isnan(report.kappa) else f"{report.kappa:.3f}"
    logger.info(
        "Shared=%d | agree=%d (%.1f%%) | Cohen's kappa=%s | target %.2f -> %s",
        report.n_shared,
        report.n_agree,
        report.percent_agree * 100,
        kappa_str,
        KAPPA_TARGET,
        "PASS" if report.meets_target else "BELOW TARGET — review guidelines",
    )
    if report.disagreements:
        logger.info("Disagreements to resolve in weekly sync (§7):")
        for d in report.disagreements:
            logger.info("  %s: A=%s B=%s | %s", d.id, d.label_a, d.label_b, d.subject[:60])


if __name__ == "__main__":
    _main()
