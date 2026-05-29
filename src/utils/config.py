"""Typed loading of training configs (CLAUDE.md §6: configs over code).

Required fields per CLAUDE.md §6: ``model_name``, ``dataset``,
``hyperparameters``, ``seed``, ``output_dir``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrainingConfig:
    """In-memory view of a YAML training config.

    Kept deliberately permissive (``extra`` catch-all) so research configs can
    carry experiment-specific keys without a schema change.
    """

    model_name: str
    dataset: str
    hyperparameters: dict[str, Any]
    seed: int
    output_dir: str
    # Optional, commonly present:
    model_type: str = "priority"  # "priority" | "phishing"
    extra: dict[str, Any] = field(default_factory=dict)

    _REQUIRED = ("model_name", "dataset", "hyperparameters", "seed", "output_dir")

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
        import yaml

        raw = yaml.safe_load(Path(path).read_text())
        missing = [k for k in cls._REQUIRED if k not in raw]
        if missing:
            raise ValueError(f"Config {path} missing required fields: {missing}")

        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        extra = {k: v for k, v in raw.items() if k not in known}
        return cls(
            model_name=raw["model_name"],
            dataset=raw["dataset"],
            hyperparameters=raw["hyperparameters"],
            seed=raw["seed"],
            output_dir=raw["output_dir"],
            model_type=raw.get("model_type", "priority"),
            extra=extra,
        )
