"""Layer 3 — Primary Signal Model / SIDE (scaffold).

Predicts **direction only** (long / short / flat) and is tuned for **high
recall** — its job is to surface every plausible opportunity; precision is
delegated downstream to the Layer 5 meta-model.

Backbone: a **CNN + LightGBM/CatBoost ensemble**, selected **regime-conditionally**
by Layer 2 (momentum-leaning logic in trend regimes, mean-reversion-leaning in
ranging regimes). Roadmap v2 can add PatchTST / SSM members weighted by regime.

The training target comes from Layer 4 direction labels; the meta-model (L5) is
later trained on this model's **out-of-fold** predictions — never in-sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..enums import Regime, Side


@dataclass
class PrimaryConfig:
    """Per-regime member weights and recall-oriented decision threshold."""

    # gbm + cnn member weights, optionally overridden per regime
    member_weights: dict[str, float] = field(default_factory=lambda: {"gbm": 0.6, "cnn": 0.4})
    regime_overrides: dict[Regime, dict[str, float]] = field(default_factory=dict)
    flat_threshold: float = 0.15  # |P_up − P_down| below this → FLAT (high recall)


class PrimarySignalModel:
    """Regime-conditional direction ensemble (SIDE)."""

    def __init__(self, config: PrimaryConfig | None = None) -> None:
        self.config = config or PrimaryConfig()
        self._members: dict[str, object] = {}  # name → fitted estimator

    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        *,
        sample_weight: pd.Series | None = None,
        regimes: pd.Series | None = None,
    ) -> PrimarySignalModel:
        """Fit ensemble members (optionally per regime). TODO(v1)."""
        raise NotImplementedError("Primary ensemble training lands in Roadmap v1 (L3).")

    def predict_proba(
        self, features: pd.DataFrame, *, regimes: pd.Series | None = None
    ) -> pd.DataFrame:
        """Return regime-weighted class probabilities [P_down, P_flat, P_up]. TODO(v1)."""
        raise NotImplementedError("Primary ensemble inference lands in Roadmap v1 (L3).")

    def predict_side(
        self, features: pd.DataFrame, *, regimes: pd.Series | None = None
    ) -> pd.Series[Side]:
        """Map probabilities to a :class:`Side` using ``flat_threshold``. TODO(v1)."""
        raise NotImplementedError("Primary side decision lands in Roadmap v1 (L3).")
