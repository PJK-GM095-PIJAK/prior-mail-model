"""Global seeding for reproducibility (CLAUDE.md §10, ML_PIPELINE.md §8).

Seeds ``random``, ``numpy``, ``torch``, and ``transformers`` from one entry point.
Note: full determinism on GPU is not guaranteed — the target is
"same seed, same machine = same result" (CLAUDE.md §10).
"""

from __future__ import annotations

import logging
import random

logger = logging.getLogger(__name__)


def set_global_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed every RNG used in a training/eval run.

    Args:
        seed: the integer seed recorded in the run config + wandb.
        deterministic: if True, ask torch for deterministic cuDNN kernels
            (slower, but improves run-to-run comparability).
    """
    random.seed(seed)

    import numpy as np

    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # transformers has its own helper that also covers its internal RNG usage.
    from transformers import set_seed as hf_set_seed

    hf_set_seed(seed)

    logger.info("Global seed set to %d (deterministic=%s)", seed, deterministic)
