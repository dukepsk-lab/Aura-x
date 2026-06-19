"""Layer 2 — Regime Detection (HMM + Hurst/KER gating).

The forecasting research's real lesson — *match the model to the signal's
character* — becomes a regime **router** here, not one monolithic model:

* a **Gaussian HMM** classifies the latent state from regime-context features;
* learned states are mapped to ``trend`` / ``range`` / ``shock`` by their
  volatility and trend-memory signatures;
* a fast **Hurst/KER/ATR-percentile gate** is the rule overlay, and the
  **shock override** (ATR-percentile breach → ``SHOCK``) is a hard safety gate —
  the single biggest protection against the one-sided-trend blow-up, letting the
  system stand down entirely.

Output is a regime label + probabilities that condition Layer 3 (the ensemble
selects per-regime member weights) and Layer 6 (sizing / stand-down). The regime
string values match the ``primary.regime_overrides`` keys in config, so
``regime_series`` feeds straight into ``PrimarySignalModel.predict_proba``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..enums import Regime
from ..types import RegimeState
from .hmm import GaussianHMM

_REGIME_COLS = ["p_trend", "p_range", "p_shock"]
_REGIME_OF_COL = {"p_trend": Regime.TREND, "p_range": Regime.RANGE, "p_shock": Regime.SHOCK}


@dataclass
class RegimeConfig:
    """Regime thresholds, the HMM feature set, and state-mapping roles."""

    hurst_trend_threshold: float = 0.55
    ker_trend_threshold: float = 0.30
    shock_atr_percentile: float = 0.90
    n_states: int = 3
    n_iter: int = 100
    seed: int = 0
    shock_override: bool = True
    # L1 feature columns used by the HMM and the gate.
    hmm_features: list[str] = field(
        default_factory=lambda: ["vol_realized_vol", "tm_hurst", "tm_ker", "vol_atr_pct"]
    )
    vol_feature: str = "vol_atr_pct"  # state→regime mapping: volatility signature
    trend_feature: str = "tm_hurst"   # state→regime mapping: trend-memory signature
    hurst_col: str = "tm_hurst"
    ker_col: str = "tm_ker"
    atr_pct_col: str = "vol_atr_pct"

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> RegimeConfig:
        r = params.get("regime", {})
        kwargs: dict[str, Any] = {
            "hurst_trend_threshold": r.get("hurst_trend_threshold", 0.55),
            "ker_trend_threshold": r.get("ker_trend_threshold", 0.30),
            "shock_atr_percentile": r.get("shock_atr_percentile", 0.90),
            "n_states": r.get("hmm_states", 3),
        }
        if "features" in r:
            kwargs["hmm_features"] = list(r["features"])
        return cls(**kwargs)


class RegimeDetector:
    """HMM + Hurst/KER/ATR-percentile regime router."""

    def __init__(self, config: RegimeConfig | None = None) -> None:
        self.config = config or RegimeConfig()
        self._hmm: GaussianHMM | None = None
        self._state_to_regime: dict[int, Regime] = {}

    # --- rule gate (fully implemented, no model required) --------------------
    def gate(self, hurst: float, ker: float, atr_percentile: float | None) -> Regime:
        """Fast rule overlay. Shock dominates (stand-down priority); else trend if
        both memory signals agree; else range."""
        cfg = self.config
        if atr_percentile is not None and atr_percentile >= cfg.shock_atr_percentile:
            return Regime.SHOCK
        if hurst >= cfg.hurst_trend_threshold and ker >= cfg.ker_trend_threshold:
            return Regime.TREND
        return Regime.RANGE

    def predict_rules(self, features: pd.DataFrame) -> pd.Series:
        """Vectorised rule gate over a feature frame (no fit needed)."""
        cfg = self.config
        hurst = features.get(cfg.hurst_col)
        ker = features.get(cfg.ker_col)
        atr = features.get(cfg.atr_pct_col)
        regime = np.full(len(features), Regime.RANGE.value, dtype=object)
        if hurst is not None and ker is not None:
            trend = (hurst.to_numpy() >= cfg.hurst_trend_threshold) & (
                ker.to_numpy() >= cfg.ker_trend_threshold
            )
            regime[trend] = Regime.TREND.value
        if atr is not None:
            regime[atr.to_numpy() >= cfg.shock_atr_percentile] = Regime.SHOCK.value
        return pd.Series(regime, index=features.index, name="regime")

    # --- HMM ----------------------------------------------------------------
    def _hmm_frame(self, features: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.config.hmm_features if c in features.columns]
        if not cols:
            raise ValueError(
                f"none of the configured HMM features {self.config.hmm_features} are present"
            )
        return features[cols]

    def fit(self, features: pd.DataFrame) -> RegimeDetector:
        """Fit the HMM on regime-context features and learn the state→regime map."""
        frame = self._hmm_frame(features).dropna()
        if len(frame) < self.config.n_states * 5:
            raise ValueError("insufficient non-NaN rows to fit the regime HMM")
        self._hmm = GaussianHMM(
            n_states=self.config.n_states, seed=self.config.seed, n_iter=self.config.n_iter
        ).fit(frame.to_numpy())
        states = self._hmm.predict(frame.to_numpy())
        self._state_to_regime = self._learn_mapping(features.loc[frame.index], states)
        return self

    def _learn_mapping(self, features: pd.DataFrame, states: np.ndarray) -> dict[int, Regime]:
        """Assign each HMM state a regime by volatility / trend-memory signature.

        Highest-volatility state → ``shock``; of the rest, the higher trend-memory
        state → ``trend``, the remainder → ``range``.
        """
        cfg = self.config
        vol = features[cfg.vol_feature].to_numpy() if cfg.vol_feature in features else None
        trend = features[cfg.trend_feature].to_numpy() if cfg.trend_feature in features else None

        uniq = sorted(set(states.tolist()))
        vol_mean = {
            s: (np.nanmean(vol[states == s]) if vol is not None else 0.0) for s in uniq
        }
        trend_mean = {
            s: (np.nanmean(trend[states == s]) if trend is not None else 0.0) for s in uniq
        }

        shock_state = max(uniq, key=lambda s: vol_mean[s])
        rest = [s for s in uniq if s != shock_state]
        mapping = {shock_state: Regime.SHOCK}
        if rest:
            trend_state = max(rest, key=lambda s: trend_mean[s])
            for s in rest:
                mapping[s] = Regime.TREND if s == trend_state else Regime.RANGE
        return mapping

    # --- combined prediction -------------------------------------------------
    def predict_regimes(self, features: pd.DataFrame) -> pd.DataFrame:
        """Regime label + ``[p_trend, p_range, p_shock]`` per row (HMM + override).

        Warm-up rows (NaN HMM features) default to ``range``. The ATR-percentile
        shock override promotes rows to ``shock`` regardless of the HMM.
        """
        if self._hmm is None:
            raise RuntimeError("RegimeDetector is not fitted; call fit() first")

        out = pd.DataFrame(0.0, index=features.index, columns=[*_REGIME_COLS])
        out["regime"] = Regime.RANGE.value
        out["p_range"] = 1.0

        frame = self._hmm_frame(features).dropna()
        if not frame.empty:
            proba = self._hmm.predict_proba(frame.to_numpy())  # (n, K)
            states = self._hmm.decode(frame.to_numpy())        # Viterbi labels
            # collapse state probabilities into regime probabilities
            reg_proba = pd.DataFrame(0.0, index=frame.index, columns=[*_REGIME_COLS])
            for state, regime in self._state_to_regime.items():
                reg_proba[f"p_{regime.value}"] += proba[:, state]
            out.loc[frame.index, _REGIME_COLS] = reg_proba.to_numpy()
            labels = [self._state_to_regime.get(int(s), Regime.RANGE).value for s in states]
            out.loc[frame.index, "regime"] = labels

        if self.config.shock_override and self.config.atr_pct_col in features.columns:
            atr = features[self.config.atr_pct_col]
            shock = atr.to_numpy() >= self.config.shock_atr_percentile
            out.loc[shock, "regime"] = Regime.SHOCK.value
        return out[["regime", *_REGIME_COLS]]

    def regime_series(self, features: pd.DataFrame) -> pd.Series:
        """Just the regime labels — feeds ``PrimarySignalModel.predict_proba``."""
        return self.predict_regimes(features)["regime"]

    def predict(self, features: pd.DataFrame) -> RegimeState:
        """Latest-bar regime + probabilities (live inference)."""
        row = self.predict_regimes(features).iloc[-1]
        cfg = self.config
        last = features.iloc[-1]
        return RegimeState(
            regime=Regime(row["regime"]),
            probabilities={_REGIME_OF_COL[c]: float(row[c]) for c in _REGIME_COLS},
            hurst=float(last.get(cfg.hurst_col, np.nan)),
            ker=float(last.get(cfg.ker_col, np.nan)),
            atr_percentile=float(last.get(cfg.atr_pct_col, np.nan)),
        )
