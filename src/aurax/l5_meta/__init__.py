"""Layer 5 — Meta-Label Model (SIZE / TRUST).

Predicts ``P(primary signal correct)`` and gates trades at ``P ≥ τ`` — the
"predict whether to *act*, not just direction" core, tuned for high precision.

* :class:`MetaLabelModel` / :class:`MetaConfig`  calibrated meta-learner + gate
* :class:`LogisticMeta`  dependency-free binary base learner
* calibrators: :class:`IsotonicCalibrator`, :class:`SigmoidCalibrator`
* :func:`build_meta_features`  primary confidence + regime + vol + recent perf
* :func:`train_meta_labeler`  fit on the primary's OOF predictions (+ diagnostics)
* :class:`MetaGatedPrimary`  primary + meta gate as one validation-ready estimator
"""

from __future__ import annotations

from .calibration import (
    IdentityCalibrator,
    IsotonicCalibrator,
    SigmoidCalibrator,
    make_calibrator,
)
from .features import build_meta_features
from .meta_model import LogisticMeta, MetaConfig, MetaLabelModel
from .pipeline import MetaGatedPrimary, meta_labels_from_sides, train_meta_labeler

__all__ = [
    "MetaLabelModel",
    "MetaConfig",
    "LogisticMeta",
    "IsotonicCalibrator",
    "SigmoidCalibrator",
    "IdentityCalibrator",
    "make_calibrator",
    "build_meta_features",
    "train_meta_labeler",
    "MetaGatedPrimary",
    "meta_labels_from_sides",
]
