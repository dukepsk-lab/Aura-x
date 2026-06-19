"""Layer 8 — Model-decay detection.

Rolling cost-adjusted Sharpe and meta-model calibration drift drive automatic
retraining triggers. The metrics below are pure/testable; the scheduler that
acts on them (and pulls fresh trades from the journal) is the v1 build target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def rolling_sharpe(returns: pd.Series, window: int, periods_per_year: int = 1512) -> pd.Series:
    """Annualised rolling Sharpe of (ideally cost-adjusted) per-bar returns."""
    mean = returns.rolling(window).mean()
    std = returns.rolling(window).std(ddof=1)
    return (mean / std.replace(0.0, np.nan)) * np.sqrt(periods_per_year)


def calibration_error(prob: np.ndarray, outcome: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error of meta-probabilities vs. realised outcomes.

    Rising ECE on recent trades is the canonical "trust layer drifting" signal.
    """
    prob = np.asarray(prob, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    if prob.size == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(prob, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        ece += mask.mean() * abs(outcome[mask].mean() - prob[mask].mean())
    return float(ece)


@dataclass
class DecayConfig:
    sharpe_window: int = 60
    min_rolling_sharpe: float = 0.0   # retrain if cost-adj Sharpe drops below
    max_calibration_error: float = 0.10


class DecayMonitor:
    """Flag retraining when edge or calibration degrades."""

    def __init__(self, config: DecayConfig | None = None) -> None:
        self.config = config or DecayConfig()

    def should_retrain(self, returns: pd.Series, prob: np.ndarray, outcome: np.ndarray) -> bool:
        cfg = self.config
        sharpe = rolling_sharpe(returns, cfg.sharpe_window).iloc[-1] if len(returns) else np.nan
        ece = calibration_error(prob, outcome)
        sharpe_bad = np.isfinite(sharpe) and sharpe < cfg.min_rolling_sharpe
        calib_bad = np.isfinite(ece) and ece > cfg.max_calibration_error
        return bool(sharpe_bad or calib_bad)
