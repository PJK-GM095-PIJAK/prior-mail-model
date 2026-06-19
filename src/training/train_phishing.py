"""Fine-tune distilbert-base-uncased for binary phishing detection.

Protocol mirrors the priority classifier (ML_PIPELINE.md section 3) with two
phishing-specific differences:
  1. Class weights favour the phishing class (``phishing_class_multiplier`` in
     config) to push recall up - kept modest in v2 (see decision log).
  2. The inference threshold is selected post-training on the validation set
     (``eval_phishing.select_threshold``), NOT the default 0.5.

Decision log:
  v1.0 (2026-06-09): bert-base-multilingual-cased, Enron-only legit, balanced
    class weights x3.0 multiplier, early stopping on recall. Passed test gates
    but over-flagged real .eml at inference.
  v2 (2026-06-18): base -> distilbert-base-uncased (English-only data, on-spec,
    faster on CPU). Class multiplier 3.0 -> 1.0 (let the threshold drive recall).
    Early stopping monitors f1_phishing, not raw recall. Model input is body-only
    (preprocess.build_phishing_input) to kill the v1 header-presence leak. Legit
    class diversified + balanced in loaders.load_phishing_dataset.
  v2.1 (2026-06-19): v2 passed §8 test gates but missed half the real .eml
    acceptance phishing (FN 0.50). Data/preprocess fixes, no hyperparameter
    change: (A1) keep URL host + email domain as signal; (B1) synthetic
    augmentation of the missed tactics (BEC, fake-invoice, credential harvest),
    header-complete on both classes, via prepare --augmentation-size; (B2) input
    back to {sender_domain} [SEP] {subject} [SEP] {body}, leak-safe because the
    augmentation supplies headers on both classes. Acceptance set is now a gate.

Run via:  make train-phishing config=configs/phishing_v2.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from src.utils.config import TrainingConfig
from src.utils.constants import (
    MAX_SEQ_LENGTH,
    PHISHING_ID2LABEL,
    PHISHING_LABEL2ID,
    PHISHING_LABELS,
)
from src.utils.repro import get_git_sha, is_working_tree_dirty
from src.utils.seeding import set_global_seed

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed/phishing")
INPUT_COLUMN = "model_input"
LABEL_COLUMN = "labels"


def _compute_metrics(eval_pred):
    """Recall, precision, and F1 for the phishing class (positive class = index 1).

    Uses argmax (â‰¡ 0.5 threshold) during training â€” only for monitoring.
    The real threshold is selected post-training in ``eval_phishing.select_threshold``.
    """
    import numpy as np
    from sklearn.metrics import f1_score, precision_score, recall_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "recall_phishing": float(
            recall_score(labels, preds, pos_label=1, zero_division=0)
        ),
        "precision_phishing": float(
            precision_score(labels, preds, pos_label=1, zero_division=0)
        ),
        "f1_phishing": float(
            f1_score(labels, preds, pos_label=1, zero_division=0)
        ),
    }


def _class_weights(train_labels, num_labels: int, phishing_multiplier: float = 3.0):
    """Inverse-frequency weights with an extra multiplier on the phishing class.

    Args:
        train_labels: array-like of integer class ids from the training split.
        num_labels: total number of classes (2 for phishing).
        phishing_multiplier: additional weight boost on phishing class (index 1).
            Raises recall at the cost of precision â€” tune via config.
    """
    import numpy as np
    import torch

    counts = np.bincount(train_labels, minlength=num_labels)
    # Inverse-frequency baseline: weight_c = N / (K * count_c).
    weights = counts.sum() / (num_labels * np.maximum(counts, 1))
    # Extra multiplier on phishing class (index 1) to push recall.
    weights[PHISHING_LABEL2ID["phishing"]] *= phishing_multiplier
    logger.info(
        "Class weights: legit=%.3f phishing=%.3f (multiplier=%.1f)",
        weights[0], weights[1], phishing_multiplier,
    )
    return torch.tensor(weights, dtype=torch.float32)


def train(config: TrainingConfig) -> None:
    """Run a phishing-detector fine-tuning job per ML_PIPELINE.md Â§3."""
    import numpy as np
    import torch
    from datasets import load_from_disk
    from torch import nn
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    hp = config.hyperparameters
    set_global_seed(config.seed)
    sha = get_git_sha()
    if is_working_tree_dirty():
        logger.warning("Working tree is dirty â€” run is not cleanly reproducible (Â§8).")
    logger.info(
        "Starting phishing training | model=%s seed=%d git=%s",
        config.model_name, config.seed, sha,
    )

    # --- Data ---------------------------------------------------------------
    if not PROCESSED_DIR.exists():
        raise FileNotFoundError(
            f"{PROCESSED_DIR} not found â€” run `make data-phishing` first."
        )
    ds = load_from_disk(str(PROCESSED_DIR))

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    max_len = int(hp.get("max_seq_length", MAX_SEQ_LENGTH))

    def _tokenize(batch):
        return tokenizer(batch[INPUT_COLUMN], truncation=True, max_length=max_len)

    keep = {INPUT_COLUMN, LABEL_COLUMN}
    tokenized = ds.map(
        _tokenize, batched=True,
        remove_columns=[c for c in ds["train"].column_names if c not in keep],
    )
    tokenized = tokenized.remove_columns([INPUT_COLUMN])
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # --- Model --------------------------------------------------------------
    num_labels = len(PHISHING_LABELS)  # 2
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=num_labels,
        id2label=PHISHING_ID2LABEL,
        label2id=PHISHING_LABEL2ID,
    )

    # --- Weighted cross-entropy with heavy phishing weight (Â§3) -------------
    phishing_mult = float(hp.get("phishing_class_multiplier", 3.0))
    use_weights = str(hp.get("class_weights", "balanced")).lower() == "balanced"
    weight_tensor = (
        _class_weights(
            np.array(tokenized["train"][LABEL_COLUMN]),
            num_labels,
            phishing_multiplier=phishing_mult,
        )
        if use_weights
        else None
    )

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            w = weight_tensor.to(outputs.logits.device) if weight_tensor is not None else None
            loss = nn.functional.cross_entropy(outputs.logits, labels, weight=w)
            return (loss, outputs) if return_outputs else loss

    # --- Mixed precision: bf16 if supported, else fp16, else none (CPU) -----
    want = str(hp.get("mixed_precision", "bf16")).lower()
    cuda = torch.cuda.is_available()
    bf16 = cuda and want == "bf16" and torch.cuda.is_bf16_supported()
    fp16 = cuda and not bf16 and want in ("bf16", "fp16")

    args = TrainingArguments(
        output_dir=config.output_dir,
        learning_rate=float(hp.get("learning_rate", 2e-5)),
        weight_decay=float(hp.get("weight_decay", 0.01)),
        per_device_train_batch_size=int(hp.get("batch_size", 16)),
        per_device_eval_batch_size=int(hp.get("batch_size", 16)),
        gradient_accumulation_steps=int(hp.get("gradient_accumulation_steps", 1)),
        num_train_epochs=float(hp.get("num_epochs", 4)),
        warmup_ratio=float(hp.get("warmup_ratio", 0.10)),
        lr_scheduler_type=str(hp.get("lr_scheduler_type", "linear")),
        bf16=bf16,
        fp16=fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        # Monitor RECALL, not F1 â€” false negatives are the worst case (Â§3).
        metric_for_best_model=str(hp.get("early_stopping_metric", "recall_phishing")),
        greater_is_better=True,
        logging_steps=50,
        seed=config.seed,
        report_to=["wandb"],
        run_name=f"{Path(config.output_dir).name}-{sha}",
    )

    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=int(hp.get("early_stopping_patience", 2))
        )
    ]

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
        compute_metrics=_compute_metrics,
        callbacks=callbacks,
    )

    trainer.train()

    # --- Persist checkpoint + tokenizer + config (reproducibility Â§8/Â§5) ---
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))

    config_path = config.extra.get("_config_path")
    if config_path:
        provenance = (
            "\n# --- run provenance (auto-recorded) ---\n"
            f"_git_sha: {sha}\n"
            f"_git_dirty: {is_working_tree_dirty()}\n"
            f"_trained_at: {datetime.now().isoformat(timespec='seconds')}\n"
        )
        (out / "training_config.yaml").write_text(Path(config_path).read_text() + provenance)

    val_metrics = trainer.evaluate()
    (out / "val_metrics.json").write_text(json.dumps(val_metrics, indent=2))
    logger.info(
        "Done. Best-model val metrics: recall_phishing=%.3f precision_phishing=%.3f",
        val_metrics.get("eval_recall_phishing", float("nan")),
        val_metrics.get("eval_precision_phishing", float("nan")),
    )
    logger.info(
        "Next step: run `make eval-phishing config=%s` to select threshold + check gates.",
        config_path or config.output_dir,
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description="Train the phishing detector")
    parser.add_argument("--config", required=True, help="path to a YAML config in configs/")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = TrainingConfig.from_yaml(args.config)
    cfg.extra["_config_path"] = args.config
    train(cfg)


if __name__ == "__main__":
    _main()

