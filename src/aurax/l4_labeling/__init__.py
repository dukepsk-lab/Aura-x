"""Layer 4 — Labeling (TRAINING-TIME ONLY).

* :func:`triple_barrier_labels`  ATR-scaled triple-barrier (primary or meta mode)
* :func:`sample_weights`         sample-uniqueness + time-decay weights
* :func:`trend_scanning_labels`  optional max-|t| trend target
* :class:`Labeler` / :class:`LabelConfig`  orchestration from YAML params

Triple-Barrier and these weights exist **only** in training; nothing here runs
at inference time.
"""

from __future__ import annotations

from .labeler import LabelConfig, Labeler
from .trend_scanning import trend_scanning_labels
from .triple_barrier import triple_barrier_labels
from .uniqueness import (
    average_uniqueness,
    num_concurrent_events,
    sample_weights,
    time_decay,
)

__all__ = [
    "Labeler",
    "LabelConfig",
    "triple_barrier_labels",
    "trend_scanning_labels",
    "sample_weights",
    "average_uniqueness",
    "num_concurrent_events",
    "time_decay",
]
