"""Fine-tune IndoBERT for 4-class email priority classification.

Driven entirely by a YAML config (CLAUDE.md §6). Protocol locked in
ML_PIPELINE.md §2: AdamW lr 2e-5, wd 0.01, linear warmup 10% + decay,
batch 16, 3–5 epochs with early stopping on val macro F1, weighted
cross-entropy, bf16/fp16, global seed.

Consumes the processed dataset produced by ``make data``
(``data/processed/priority``: train/validation/test with ``model_input`` +
``labels``). Run via ``make train config=configs/priority_baseline.yaml``.

NOTE: a real run needs GPU + confirmed budget (CLAUDE.md §3/§11). On CPU this
will technically run but is far too slow for the full dataset.
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
    PRIORITY_ID2LABEL,
    PRIORITY_LABEL2ID,
    PRIORITY_LABELS,
)
from src.utils.repro import get_git_sha, is_working_tree_dirty
from src.utils.seeding import set_global_seed

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed/priority")
INPUT_COLUMN = "model_input"
LABEL_COLUMN = "labels"


def _compute_metrics(eval_pred):
    """Macro F1 (the early-stopping + promotion metric) + per-class recall."""
    import numpy as np
    from sklearn.metrics import f1_score, recall_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    per_class_recall = recall_score(
        labels, preds, labels=list(range(len(PRIORITY_LABELS))), average=None, zero_division=0
    )
    metrics = {"macro_f1": f1_score(labels, preds, average="macro", zero_division=0)}
    for i, name in enumerate(PRIORITY_LABELS):
        metrics[f"recall_{name}"] = float(per_class_recall[i])
    return metrics


def _class_weights(train_labels, num_labels: int):
    """Inverse-frequency class weights from the TRAIN distribution (§2)."""
    import numpy as np
    import torch

    counts = np.bincount(train_labels, minlength=num_labels)
    # weight_c = N / (K * count_c); normalized so equal counts -> all-ones.
    weights = counts.sum() / (num_labels * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def train(config: TrainingConfig) -> None:
    """Run a priority-classifier fine-tuning job per ML_PIPELINE.md §2."""
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
        logger.warning("Working tree is dirty — run is not cleanly reproducible (§8).")
    logger.info("Starting priority training | model=%s seed=%d git=%s", config.model_name, config.seed, sha)

    # --- Data ---------------------------------------------------------------
    if not PROCESSED_DIR.exists():
        raise FileNotFoundError(f"{PROCESSED_DIR} not found — run `make data` first.")
    ds = load_from_disk(str(PROCESSED_DIR))

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    max_len = int(hp.get("max_seq_length", MAX_SEQ_LENGTH))

    def _tokenize(batch):
        # Truncate from the end to fit the 512-token budget (subject is prepended) — §2.
        return tokenizer(batch[INPUT_COLUMN], truncation=True, max_length=max_len)

    keep = {INPUT_COLUMN, LABEL_COLUMN}
    tokenized = ds.map(
        _tokenize, batched=True, remove_columns=[c for c in ds["train"].column_names if c not in keep]
    )
    tokenized = tokenized.remove_columns([INPUT_COLUMN])
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # --- Model --------------------------------------------------------------
    num_labels = len(PRIORITY_LABELS)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=num_labels,
        id2label=PRIORITY_ID2LABEL,
        label2id=PRIORITY_LABEL2ID,
    )

    # --- Weighted cross-entropy (class imbalance, §2) -----------------------
    use_weights = str(hp.get("class_weights", "balanced")).lower() == "balanced"
    weight_tensor = _class_weights(np.array(tokenized["train"][LABEL_COLUMN]), num_labels) if use_weights else None

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            w = weight_tensor.to(outputs.logits.device) if weight_tensor is not None else None
            loss = nn.functional.cross_entropy(outputs.logits, labels, weight=w)
            return (loss, outputs) if return_outputs else loss

    # --- Precision: bf16 if supported, else fp16, else neither (CPU) --------
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
        metric_for_best_model=str(hp.get("early_stopping_metric", "macro_f1")),
        greater_is_better=True,
        logging_steps=50,
        seed=config.seed,
        report_to=["wandb"],
        run_name=f"priority-baseline-{sha}",
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=int(hp.get("early_stopping_patience", 2)))]

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

    # --- Persist checkpoint + tokenizer + config (reproducibility, §8/§5) ---
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    # Copy the exact config next to the checkpoint (reproducibility, §5/§8).
    config_path = config.extra.get("_config_path")
    if config_path:
        # Copy the config and append a provenance block so the run is always
        # reproducible (ML_PIPELINE.md §8) — no manual SHA lookup needed later.
        provenance = (
            "\n# --- run provenance (auto-recorded) ---\n"
            f"_git_sha: {sha}\n"
            f"_git_dirty: {is_working_tree_dirty()}\n"
            f"_trained_at: {datetime.now().isoformat(timespec='seconds')}\n"
        )
        (out / "training_config.yaml").write_text(Path(config_path).read_text() + provenance)
    val_metrics = trainer.evaluate()
    (out / "val_metrics.json").write_text(json.dumps(val_metrics, indent=2))
    logger.info("Done. Best-model val metrics: %s", val_metrics)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Train the priority classifier")
    parser.add_argument("--config", required=True, help="path to a YAML config in configs/")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = TrainingConfig.from_yaml(args.config)
    cfg.extra["_config_path"] = args.config  # so we can copy it next to the checkpoint
    train(cfg)


if __name__ == "__main__":
    _main()
