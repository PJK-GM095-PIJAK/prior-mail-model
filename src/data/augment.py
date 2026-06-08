"""Optional data augmentation for the email-priority domain.

Not in the locked training protocol yet — kept as a stub so augmentation
experiments have a home. Any augmentation used in a promoted run must be
recorded in the training config and model card.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def augment(dataset, seed: int):
    """Return an augmented copy of ``dataset``. Stub — no augmentation by default."""
    raise NotImplementedError("augment: no augmentation strategy decided yet.")
