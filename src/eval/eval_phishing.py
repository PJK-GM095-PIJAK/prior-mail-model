"""Evaluation harness for the phishing detector.

Eval gates (ML_PIPELINE.md §3, CLAUDE.md §8):
  - Recall    >= 0.95 on test set  (false negatives are the worst case)
  - Precision >= 0.80 on test set
  - Inference latency p95 < 500 ms per email on CPU

The decision threshold is chosen on the VALIDATION set (never the test set) to
achieve recall >= 0.95 while maximising precision, then persisted as
``threshold.json``.  Adversarial checks (Faiz) run separately via
``--adversarial`` and write to ``eval/results/adversarial/``.

Run via:  make eval-phishing config=configs/phishing_v2.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.eval.benchmarks import measure_latency_ms
from src.utils.config import TrainingConfig
from src.utils.constants import MAX_SEQ_LENGTH

logger = logging.getLogger(__name__)

# Promotion thresholds — do not relax without team sign-off (CLAUDE.md §8).
RECALL_GATE: float = 0.95
PRECISION_GATE: float = 0.80
LATENCY_P95_MS_GATE: float = 500.0

PROCESSED_DIR = Path("data/processed/phishing")
RESULTS_DIR = Path("eval/results/phishing")
ADVERSARIAL_RESULTS_DIR = Path("eval/results/adversarial")
INPUT_COLUMN = "model_input"
LABEL_COLUMN = "labels"


# ---------------------------------------------------------------------------
# Pure gate logic — unit-testable without a checkpoint
# ---------------------------------------------------------------------------

def check_gates(metrics: dict) -> dict:
    """Decide pass/fail per gate from computed metrics.

    Separated from model code so it is unit-testable without a GPU.
    Expects keys: ``recall_phishing``, ``precision_phishing``, ``p95_ms``.
    """
    gates = {
        "recall": metrics["recall_phishing"] >= RECALL_GATE,
        "precision": metrics["precision_phishing"] >= PRECISION_GATE,
        "latency_p95": metrics["p95_ms"] < LATENCY_P95_MS_GATE,
    }
    return {
        "gates": gates,
        "all_passed": all(gates.values()),
    }


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------

# Recall is weighted beta× precision in threshold selection (F-beta). beta=2 →
# the chosen threshold leans toward catching phishing (FN-averse, CLAUDE.md §8).
THRESHOLD_BETA_SQ: float = 4.0  # beta = 2  → beta**2 = 4


def select_threshold(val_scores, val_labels) -> float:
    """Pick an FN-averse decision threshold on the VALIDATION set.

    False negatives are the worst case for a phishing detector (CLAUDE.md §8).
    The earlier rule maximised *precision* subject to recall >= 0.95, which picks
    the HIGHEST passing threshold. Once the synthetic augmentation polarised the
    val scores, val-recall stayed >= 0.95 all the way up to ~0.95, so the rule
    pushed the threshold to ~0.95 and silently cut borderline REAL phishing
    (acceptance ``it_support`` scored 0.89 and ``package_delivery`` 0.63 — both
    detected as suspicious, both dropped by the 0.95 cut).

    We now maximise F-beta (beta=2, recall weighted 2× precision) among thresholds
    that meet BOTH gates (recall >= 0.95 AND precision >= 0.80) on val — favouring
    recall without abandoning the precision contract. Fallbacks, in order: the
    highest-recall threshold still meeting precision >= 0.80, then 0.5.

    The search is done on the VALIDATION set; the test set is never touched
    during threshold selection (CLAUDE.md §8).

    Args:
        val_scores: iterable of float phishing-class probabilities (0–1).
        val_labels: iterable of int labels (0 = legit, 1 = phishing).

    Returns:
        A float threshold in [0, 1].
    """
    import numpy as np
    from sklearn.metrics import precision_score, recall_score

    scores = np.asarray(val_scores, dtype=float)
    labels = np.asarray(val_labels, dtype=int)

    candidates: list[tuple[float, float]] = []  # (fbeta, threshold) meeting both gates
    # Fallback: highest recall among thresholds that still satisfy the precision gate.
    fb_threshold = 0.5
    fb_recall = -1.0

    for t in np.arange(0.05, 0.96, 0.01):
        preds = (scores >= t).astype(int)
        recall = float(recall_score(labels, preds, pos_label=1, zero_division=0))
        precision = float(precision_score(labels, preds, pos_label=1, zero_division=0))

        if precision >= PRECISION_GATE and recall > fb_recall:
            fb_recall = recall
            fb_threshold = float(round(t, 4))

        if recall >= RECALL_GATE and precision >= PRECISION_GATE:
            denom = THRESHOLD_BETA_SQ * precision + recall
            fbeta = (1 + THRESHOLD_BETA_SQ) * precision * recall / denom if denom > 0 else 0.0
            candidates.append((fbeta, float(round(t, 4))))

    if candidates:
        # On a realistic (overlapping) val set F2 has a single interior peak. On a
        # near-separable set many thresholds tie at the top — pick the MEDIAN of
        # that optimal plateau: the most robust operating point, avoiding both the
        # over-flagging low edge and the FN-prone high edge.
        best_fbeta = max(f for f, _ in candidates)
        tied = [t for f, t in candidates if f >= best_fbeta - 1e-9]
        best_threshold = float(np.median(tied))
        final_preds = (scores >= best_threshold).astype(int)
        logger.info(
            "Selected threshold=%.4f (F2-max plateau median, both gates met) | "
            "val recall=%.3f precision=%.3f",
            best_threshold,
            float(recall_score(labels, final_preds, pos_label=1, zero_division=0)),
            float(precision_score(labels, final_preds, pos_label=1, zero_division=0)),
        )
        return best_threshold

    if fb_recall >= 0.0:
        logger.warning(
            "No threshold met both gates on val; falling back to highest-recall "
            "threshold=%.4f (precision >= %.2f, val recall=%.3f). Gates may fail.",
            fb_threshold, PRECISION_GATE, fb_recall,
        )
        return fb_threshold

    logger.warning(
        "No threshold met precision >= %.2f on val; falling back to 0.5 — model "
        "likely needs more training or data. Gates will fail.",
        PRECISION_GATE,
    )
    return 0.5


# ---------------------------------------------------------------------------
# Full evaluation harness
# ---------------------------------------------------------------------------

def evaluate(config: TrainingConfig, *, adversarial_dir: Path | None = None) -> dict:
    """Evaluate checkpoint: select threshold, check gates, save artefacts.

    Steps:
    1. Load model + tokenizer from ``config.output_dir``.
    2. Run inference on the VALIDATION split to select the threshold.
    3. Persist ``threshold.json`` next to the checkpoint.
    4. Run inference on the TEST split using the selected threshold.
    5. Compute recall, precision, F1, confusion matrix.
    6. Measure CPU latency (p95).
    7. Run ``check_gates``; write ``eval_report.json`` to ``RESULTS_DIR``.
    8. Optionally evaluate on the adversarial set (``adversarial_dir``).

    Args:
        config: training config pointing at the checkpoint to evaluate.
        adversarial_dir: if given, a directory of JSON files with the same
            schema as the processed dataset (``model_input``, ``labels``).
            Results are written to ``ADVERSARIAL_RESULTS_DIR``.

    Returns:
        The full eval report dict (same content as ``eval_report.json``).
    """
    import torch
    from datasets import load_from_disk
    from sklearn.metrics import (
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ckpt = Path(config.output_dir)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt} — run `make train-phishing` first."
        )
    if not PROCESSED_DIR.exists():
        raise FileNotFoundError(
            f"{PROCESSED_DIR} not found — run `make data-phishing` first."
        )

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    model = AutoModelForSequenceClassification.from_pretrained(str(ckpt))
    model.eval()
    model.to("cpu")  # gate is measured on CPU (Render target, §3)
    max_len = int(config.hyperparameters.get("max_seq_length", MAX_SEQ_LENGTH))

    @torch.no_grad()
    def _phishing_prob(text: str) -> float:
        """Return phishing class probability for one email input string."""
        enc = tokenizer(text, truncation=True, max_length=max_len, return_tensors="pt")
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        return float(probs[0, 1].item())  # phishing = index 1

    ds = load_from_disk(str(PROCESSED_DIR))

    # --- Step 2: threshold selection on VALIDATION set ----------------------
    val_texts = ds["validation"][INPUT_COLUMN]
    val_labels = ds["validation"][LABEL_COLUMN]
    logger.info("Running inference on %d validation examples for threshold selection…", len(val_texts))
    val_scores = [_phishing_prob(t) for t in val_texts]
    threshold = select_threshold(val_scores, val_labels)

    # --- Step 3: persist threshold.json ------------------------------------
    threshold_data = {"threshold": threshold}
    (ckpt / "threshold.json").write_text(json.dumps(threshold_data, indent=2))
    logger.info("Saved threshold.json → %s", ckpt / "threshold.json")

    # --- Step 4–5: evaluate on TEST set using selected threshold ------------
    test_texts = ds["test"][INPUT_COLUMN]
    y_true = list(ds["test"][LABEL_COLUMN])
    logger.info("Running inference on %d test examples…", len(test_texts))
    test_scores = [_phishing_prob(t) for t in test_texts]
    y_pred = [1 if s >= threshold else 0 for s in test_scores]

    metrics: dict = {
        "recall_phishing": float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "precision_phishing": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "f1_phishing": float(
            f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "threshold": threshold,
        "n_test": len(y_true),
        "n_phishing_test": int(sum(1 for l in y_true if l == 1)),
        "n_legit_test": int(sum(1 for l in y_true if l == 0)),
    }

    # --- Step 6: CPU latency (p95 on capped sample) -------------------------
    latency = measure_latency_ms(
        _phishing_prob, test_texts[: min(len(test_texts), 200)]
    )
    metrics["p95_ms"] = latency["p95_ms"]
    metrics["latency"] = latency

    # --- Step 7: gate check + save report -----------------------------------
    gate_result = check_gates(metrics)
    report = {
        "checkpoint": str(ckpt),
        "test_split": str(PROCESSED_DIR / "test"),
        "metrics": metrics,
        **gate_result,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "eval_report.json").write_text(json.dumps(report, indent=2))
    logger.info(
        "Gates %s | recall=%.3f precision=%.3f p95=%.0fms threshold=%.4f",
        "PASSED ✅" if gate_result["all_passed"] else "FAILED ❌",
        metrics["recall_phishing"],
        metrics["precision_phishing"],
        metrics["p95_ms"],
        threshold,
    )
    if not gate_result["all_passed"]:
        failed = [k for k, v in gate_result["gates"].items() if not v]
        logger.warning("Failed gates: %s", failed)

    # --- Step 8 (optional): adversarial set ---------------------------------
    if adversarial_dir is not None:
        _evaluate_adversarial(
            adversarial_dir, _phishing_prob, threshold, config
        )

    # Best-effort wandb logging.
    try:
        import wandb

        if wandb.run is not None:
            wandb.log({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
    except Exception as e:
        logger.debug("wandb logging skipped: %s", e)

    return report


def _evaluate_adversarial(
    adv_dir: Path,
    predict_fn,
    threshold: float,
    config: TrainingConfig,
) -> None:
    """Evaluate on a directory of adversarial JSON files (Faiz's test set).

    Each file must have at minimum keys: ``model_input`` (str), ``labels`` (int).
    Results are written to ``ADVERSARIAL_RESULTS_DIR / eval_adversarial.json``.
    Adversarial performance does NOT affect gate pass/fail — it is informational.
    """
    import glob

    from sklearn.metrics import precision_score, recall_score

    files = glob.glob(str(adv_dir / "*.json"))
    if not files:
        logger.warning("No JSON files found in adversarial dir %s — skipping.", adv_dir)
        return

    adv_inputs: list[str] = []
    adv_labels: list[int] = []
    for f in sorted(files):
        data = json.loads(Path(f).read_text())
        if isinstance(data, list):
            for row in data:
                adv_inputs.append(row[INPUT_COLUMN])
                adv_labels.append(int(row[LABEL_COLUMN]))
        else:
            adv_inputs.append(data[INPUT_COLUMN])
            adv_labels.append(int(data[LABEL_COLUMN]))

    adv_scores = [predict_fn(t) for t in adv_inputs]
    adv_preds = [1 if s >= threshold else 0 for s in adv_scores]
    adv_metrics = {
        "recall_phishing": float(recall_score(adv_labels, adv_preds, pos_label=1, zero_division=0)),
        "precision_phishing": float(precision_score(adv_labels, adv_preds, pos_label=1, zero_division=0)),
        "n": len(adv_labels),
        "threshold": threshold,
    }
    ADVERSARIAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (ADVERSARIAL_RESULTS_DIR / "eval_adversarial.json").write_text(
        json.dumps({"checkpoint": str(config.output_dir), "metrics": adv_metrics}, indent=2)
    )
    logger.info(
        "Adversarial set (%d examples): recall=%.3f precision=%.3f",
        adv_metrics["n"], adv_metrics["recall_phishing"], adv_metrics["precision_phishing"],
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the phishing detector")
    parser.add_argument("--config", required=True, help="path to a YAML config in configs/")
    parser.add_argument(
        "--adversarial", default=None,
        help="path to adversarial test JSON directory (optional; Faiz's test set)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = TrainingConfig.from_yaml(args.config)
    adv_dir = Path(args.adversarial) if args.adversarial else None
    report = evaluate(cfg, adversarial_dir=adv_dir)
    raise SystemExit(0 if report["all_passed"] else 1)


if __name__ == "__main__":
    _main()
