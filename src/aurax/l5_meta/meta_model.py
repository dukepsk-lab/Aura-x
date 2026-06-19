"""Layer 5 — Meta-Label Model / TRUST (scaffold).

The "self-trust" layer. Takes the primary signal **plus** regime, volatility and
recent-performance features and predicts **P(primary signal is correct)**.

* A **calibrated stacking meta-learner** — calibration (Platt / isotonic) is
  essential so the output is usable as a *sizing input*, not just a classifier.
* Tuned for **high precision** — it filters false positives, protecting the
  thin H4 cost budget.
* A signal proceeds only if ``P(correct) ≥ τ``, with ``τ`` tuned on
  cost-adjusted, baseline-relative CV performance.

**Critical:** the meta-model must be trained on the primary model's
**out-of-fold** predictions, never in-sample (see ``aurax.validation``).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class MetaConfig:
    calibration: str = "isotonic"  # 'isotonic' | 'sigmoid' (Platt)
    threshold_tau: float = 0.55    # trade iff P(correct) ≥ τ
    cv_folds: int = 5


class MetaLabelModel:
    """Calibrated meta-learner predicting P(primary signal correct)."""

    def __init__(self, config: MetaConfig | None = None) -> None:
        self.config = config or MetaConfig()
        self._model = None  # set by fit(): calibrated stacking classifier

    def fit(
        self,
        meta_features: pd.DataFrame,
        meta_labels: pd.Series,
        *,
        sample_weight: pd.Series | None = None,
    ) -> MetaLabelModel:
        """Fit + calibrate on out-of-fold primary predictions. TODO(v1)."""
        raise NotImplementedError("Meta-model training lands in Roadmap v1 (L5).")

    def predict_proba(self, meta_features: pd.DataFrame) -> pd.Series:
        """Return calibrated P(signal correct). TODO(v1)."""
        raise NotImplementedError("Meta-model inference lands in Roadmap v1 (L5).")

    def gate(self, meta_features: pd.DataFrame) -> pd.Series:
        """Boolean trade gate: ``P(correct) ≥ τ``. TODO(v1)."""
        raise NotImplementedError("Meta gate lands in Roadmap v1 (L5).")
