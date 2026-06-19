"""Layer 2 — Regime Detection (scaffold).

The forecasting research's real lesson — *match the model to the signal's
character* — becomes a regime router here rather than one monolithic model.

* **HMM** classifies state: trending / ranging / high-volatility-shock.
* **Hurst / KER gating** is a fast confirmation overlay.
* In a **shock** regime the system can stand down entirely (the single biggest
  protection against trend-blow-up risk).

The Hurst/KER/ATR-percentile gate below is already implementable from Layer 1
features; the HMM member is the v1 build target (``fit``/``predict_proba`` raise
``NotImplementedError`` until then).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..enums import Regime
from ..types import RegimeState


@dataclass
class RegimeConfig:
    hurst_trend_threshold: float = 0.55
    ker_trend_threshold: float = 0.30
    shock_atr_percentile: float = 0.90
    hmm_states: int = 3


class RegimeDetector:
    """HMM + Hurst/KER/ATR-percentile regime router."""

    def __init__(self, config: RegimeConfig | None = None) -> None:
        self.config = config or RegimeConfig()
        self._hmm = None  # set by fit(); e.g. hmmlearn.GaussianHMM

    def gate(self, hurst: float, ker: float, atr_percentile: float) -> Regime:
        """Fast rule overlay (fully implemented, no model required).

        Shock dominates (stand-down priority); else trend if both memory signals
        agree; else range.
        """
        cfg = self.config
        if atr_percentile is not None and atr_percentile >= cfg.shock_atr_percentile:
            return Regime.SHOCK
        if hurst >= cfg.hurst_trend_threshold and ker >= cfg.ker_trend_threshold:
            return Regime.TREND
        return Regime.RANGE

    def fit(self, features: pd.DataFrame) -> RegimeDetector:
        """Fit the HMM on regime-context features. TODO(v1)."""
        raise NotImplementedError("HMM fitting lands in Roadmap v1 (L2).")

    def predict(self, features: pd.DataFrame) -> RegimeState:
        """Return the current regime + probabilities. TODO(v1)."""
        raise NotImplementedError("HMM inference lands in Roadmap v1 (L2).")
