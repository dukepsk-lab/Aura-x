"""Layer 5 — Meta-Label Model (SIZE/TRUST): calibrated stacking meta-learner."""

from __future__ import annotations

from .meta_model import MetaConfig, MetaLabelModel

__all__ = ["MetaLabelModel", "MetaConfig"]
