# PriorMail Model

The **machine learning** repo of PriorMail. It fine-tunes **DistilBERT** for two email tasks:

- **Priority classification** — 4 classes (`urgent`, `high`, `normal`, `low`), trained on the team-curated `insanar/prior-mail-priority` dataset.
- **Phishing detection** — binary (in progress).

Trained checkpoints are consumed by `prior-mail-backend` at inference time. See [CLAUDE.md](CLAUDE.md) for the full pipeline, datasets, and evaluation gates.

## Quick start

```bash
make install                                # install deps via uv
make data                                   # download + preprocess the priority dataset
make train config=configs/priority_v2.yaml  # fine-tune
make eval  config=configs/priority_v2.yaml  # run the eval gates
```

## Model Link

Phishing Model: https://huggingface.co/faizhuda/priormail-phishing
Priority Model: https://huggingface.co/insanar/priormail-priority 