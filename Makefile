# PriorMail ML — task runner. Commands mirror CLAUDE.md §13.
# Usage examples:
#   make install
#   make train config=configs/priority_baseline.yaml
#   make eval  config=configs/priority_baseline.yaml
#   make export checkpoint=checkpoints/priority_v1

.PHONY: help install data train eval export notebooks lint format typecheck test clean submodule

# Prefer the project venv automatically so you don't have to `source` it.
# Falls back to bare `python` (CI / Colab / an already-activated env).
VENV := .venv
PY := $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python)

# Allow `make train config=...` and `make export checkpoint=...`
config ?=
checkpoint ?=
# Dataset version (HF config) for `make data`. Empty -> prepare.py's default (v2).
subset ?=

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install package + dev deps via uv
	uv pip install -e ".[dev]"

data:  ## Download + preprocess the priority dataset. Optional: subset=v4
	$(PY) -m src.data.prepare $(if $(subset),--subset $(subset),)

train:  ## Run training. Requires: config=configs/<name>.yaml
	@if [ -z "$(config)" ]; then echo "ERROR: pass config=configs/<name>.yaml"; exit 1; fi
	$(PY) -m src.training.train_priority --config $(config)

eval:  ## Run eval suite. Requires: config=configs/<name>.yaml
	@if [ -z "$(config)" ]; then echo "ERROR: pass config=configs/<name>.yaml"; exit 1; fi
	$(PY) -m src.eval.eval_priority --config $(config)

export:  ## Package + upload checkpoint to Supabase. Requires: checkpoint=path/to/ckpt
	@if [ -z "$(checkpoint)" ]; then echo "ERROR: pass checkpoint=path/to/ckpt"; exit 1; fi
	$(PY) -m src.exporter.export --checkpoint $(checkpoint)

notebooks:  ## Launch Jupyter (exploration only)
	$(PY) -m jupyter lab notebooks/

lint:  ## Lint with ruff
	$(PY) -m ruff check src tests

format:  ## Format with ruff
	$(PY) -m ruff format src tests

typecheck:  ## Type-check (strict for src/exporter only)
	$(PY) -m mypy src

test:  ## Run pytest
	$(PY) -m pytest

submodule:  ## Init/update the shared-specs submodule
	git submodule update --init --recursive

clean:  ## Remove caches (NOT data/checkpoints — those are precious)
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .mypy_cache .ruff_cache .pytest_cache
