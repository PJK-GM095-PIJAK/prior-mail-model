# Backend Integration — Loading the Priority Model

> For the backend owner (Syafiq, per CLAUDE.md §15). How to load and run the
> trained priority classifier from `prior-mail-backend`. Reference contract:
> `docs/ML_PIPELINE.md` §6 and `docs/DATA_MODELS.md`.

---

## ⚠️ READ THIS FIRST — the model is on HuggingFace, not Supabase

`ML_PIPELINE.md` §6 says models live at `supabase://models/...`. **They do not,
currently.** Free-tier Supabase caps Storage uploads at 50MB; our checkpoint is
~475MB. So the models are published to the **HuggingFace Hub** instead:

| Model | Version | Base | Location |
|---|---|---|---|
| priority (English topical proxy) | v1.0 | IndoBERT | `hf://insanar/priormail-priority/v1.0` |
| priority (Indonesian) | v1.1 | IndoBERT | `hf://insanar/priormail-priority/v1.1` |
| priority (English, curated dataset) | **v2.0** | DistilBERT | _in progress — not yet published_ |

> **Migration in progress.** The repo now fine-tunes **DistilBERT** on the curated
> `insanar/prior-mail-priority` dataset (English, direct 4-class labels). That is the
> **v2.0** lineage and the target going forward — but it is **not published yet** (a
> training run is pending). Until v2.0 ships, **`v1.1` (IndoBERT) is the newest
> loadable checkpoint**; the load steps below apply unchanged when v2.0 lands.

**Storage is also an unresolved contract decision.** Three options — pick one with the team:
1. Backend loads from HF (this guide shows how — easiest, no cost), or
2. Upgrade Supabase to Pro ($25/mo) and re-upload there to honor §6 as written, or
3. Quantize the model under 50MB.

Until decided, treat the env var as e.g. `PRIORITY_MODEL_URI=hf://insanar/priormail-priority/v1.1`.

> **No version is "promoted."** The v1.x baselines are trained on a topic→priority
> *proxy* (not true urgency). Wiring them up for testing is fine; pointing production
> traffic at any version is a deliberate team call (§11).

---

## 1. What you download

The version folder contains a standard HuggingFace model directory:

```
config.json            ← has id2label/label2id — the label contract
model.safetensors      ← weights (~475MB)
tokenizer.json         ← fast tokenizer
tokenizer_config.json
eval_report.json       ← metrics (informational)
model_card.md          ← description + limitations (READ the limitations)
training_config.yaml   ← provenance (git SHA, etc.)
```

## 2. Load it (transformers loads HF repos directly)

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO = "insanar/priormail-priority"
REVISION = "main"   # or pin a commit; files are under the v1.1/ subfolder

# NOTE: files are in a `v1.1/` subfolder of the repo. Either download that
# subfolder, or publish each version as its own repo. Simplest: use
# huggingface_hub.snapshot_download(REPO, allow_patterns="v1.1/*") then load
# from the local path.
tokenizer = AutoTokenizer.from_pretrained("/local/path/to/v1.1")
model = AutoModelForSequenceClassification.from_pretrained("/local/path/to/v1.1")
model.eval()
```

Per §6: load **once** at FastAPI lifespan startup; if load fails, refuse to start
(log + exit non-zero). Expose the version string at `GET /api/v1/_health/models`.

## 3. ⚠️ Input format MUST match training exactly

This is the #1 way to silently get wrong predictions. The model was trained on a
**specific preprocessed string**, not raw email. You must reproduce it:

**Format:** `{cleaned_subject} [SEP] {cleaned_body}`

**Cleaning (in order):** strip HTML → collapse whitespace → replace URLs with
`[URL]` → replace emails with `[EMAIL]`. Then join subject and body with ` [SEP] `.

The exact logic is in this repo at `src/data/preprocess.py`
(`clean_text` + `build_priority_input`). **Port that logic to the backend** (or
copy it) — do not feed raw email text. Truncate to 512 tokens (the tokenizer does
this with `truncation=True, max_length=512`).

## 4. Run inference + map outputs to the DB fields

```python
import torch

def classify_priority(subject: str, body: str) -> tuple[str, float]:
    text = build_priority_input(subject, body)   # the §3 cleaning above
    enc = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
    probs = logits.softmax(-1)[0]
    idx = int(probs.argmax())
    return model.config.id2label[idx], float(probs[idx])
```

**Label contract** (from `config.json` — must match `priority_level` enum in
`DATA_MODELS.md`):

| id | label |
|---|---|
| 0 | `urgent` |
| 1 | `high` |
| 2 | `normal` |
| 3 | `low` |

Map to the `emails` table:
- `priority` ← the label string (`urgent`/`high`/`normal`/`low`)
- `priority_confidence` ← the softmax prob of the chosen class (0–1)
- `model_versions` ← `{"priority": "v1.1"}` (so each row records which model produced it, §6)

## 5. The Classifier protocol (§6)

Wrap it in the thin class §6 specifies so implementations are swappable:

```python
class Classifier(Protocol):
    version: str
    def predict(self, inputs: list[str]) -> list[Prediction]: ...
```

Lives in `prior-mail-backend/src/priormail/services/classifier.py`.

## 6. Performance note
Measured p95 latency is ~50ms/email on CPU (well under the 500ms budget), so it
runs fine on Render's CPU instances. Batch if you process many at once.

## 7. Honest caveats (from the model card — don't skip)
- The published **v1.x** models judge **topic→priority** (e.g. "OTP code vs promo"),
  NOT true urgency — v1.0 on English, v1.1 on Indonesian translated text.
- **v2.0** (DistilBERT) trains on the curated `insanar/prior-mail-priority` set: native
  English with direct priority labels (incl. synthetic urgency examples), but still
  leans topic→priority rather than fully validated urgency.
- Treat the output as a useful signal, not ground truth — expect to retrain on more
  real labeled data before it's production-grade.

---

*Questions on the model itself → Insan. Contract/field questions → `DATA_MODELS.md`.*
