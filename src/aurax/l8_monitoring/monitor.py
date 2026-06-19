"""Monitor — Layer 8 orchestration & lifecycle.

Ties alerts, decay detection and the trade journal together:

* **alerts** on fills, drawdown breaches, regime flips and model decay;
* **decay detection** — rolling cost-adjusted Sharpe + meta-calibration drift →
  a retrain signal;
* **dashboard snapshot** — the live metrics for the VENUS-X terminal, computed
  from the journal (which also feeds continuous CPCV refresh).

Stateless math lives in :mod:`aurax.l8_monitoring.decay`; this class holds the
small amount of live state (peak equity, last regime) and routes alerts through
an injected :class:`~aurax.l8_monitoring.alerts.Notifier`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..enums import Regime
from .alerts import AlertKind, LogNotifier, Notifier
from .decay import DecayConfig, DecayMonitor, DecayReport, calibration_error, rolling_sharpe


@dataclass
class MonitorConfig:
    decay: DecayConfig = field(default_factory=DecayConfig)
    drawdown_alert: float = 0.10
    sharpe_window: int = 60
    periods_per_year: int = 1512

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> MonitorConfig:
        risk = params.get("risk", {})
        return cls(
            decay=DecayConfig(),
            drawdown_alert=risk.get("max_drawdown_breaker", 0.10),
            periods_per_year=params.get("validation", {}).get("periods_per_year", 1512),
        )


@dataclass
class DashboardSnapshot:
    """Live monitoring metrics (the VENUS-X terminal feed)."""

    equity: float
    peak_equity: float
    drawdown: float
    rolling_sharpe: float
    hit_rate: float
    calibration_error: float
    n_trades: int
    total_pnl: float
    regime: str | None
    retrain_due: bool
    open_positions: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "equity": self.equity,
            "drawdown": self.drawdown,
            "rolling_sharpe": self.rolling_sharpe,
            "hit_rate": self.hit_rate,
            "calibration_error": self.calibration_error,
            "n_trades": self.n_trades,
            "total_pnl": self.total_pnl,
            "regime": self.regime,
            "retrain_due": self.retrain_due,
            "open_positions": self.open_positions,
        }


def journal_metrics(
    trades: pd.DataFrame | None, *, window: int = 60, periods_per_year: int = 1512
) -> dict[str, Any]:
    """Compute monitoring metrics from the trade journal (closed trades).

    Expects ``pnl_net`` (or ``pnl``) and ``meta_prob`` columns; tolerant of
    missing columns / empty frames.
    """
    empty = {
        "n_trades": 0, "total_pnl": 0.0, "hit_rate": float("nan"),
        "rolling_sharpe": float("nan"), "calibration_error": float("nan"),
    }
    if trades is None or len(trades) == 0:
        return empty
    pnl_col = "pnl_net" if "pnl_net" in trades.columns else "pnl" if "pnl" in trades.columns else None
    if pnl_col is None:
        return empty
    closed = trades[trades[pnl_col].notna()]
    n = len(closed)
    if n == 0:
        return empty

    pnl = closed[pnl_col].astype(float).reset_index(drop=True)
    won = (pnl > 0).astype(int)
    eff = max(2, min(window, n))
    rs = rolling_sharpe(pnl, eff, periods_per_year).dropna()

    ece = float("nan")
    if "meta_prob" in closed.columns:
        mp = closed["meta_prob"].astype(float).to_numpy()
        valid = np.isfinite(mp)
        if valid.any():
            ece = calibration_error(mp[valid], won.to_numpy()[valid])

    return {
        "n_trades": n,
        "total_pnl": float(pnl.sum()),
        "hit_rate": float(won.mean()),
        "rolling_sharpe": float(rs.iloc[-1]) if len(rs) else float("nan"),
        "calibration_error": ece,
    }


class Monitor:
    """Live monitoring orchestrator + lifecycle (alerts, decay, dashboard)."""

    def __init__(self, config: MonitorConfig | None = None, notifier: Notifier | None = None) -> None:
        self.config = config or MonitorConfig()
        self.notifier = notifier or LogNotifier()
        self.decay = DecayMonitor(self.config.decay)
        self._peak = -math.inf
        self._last_regime: str | None = None

    # --- event hooks ---------------------------------------------------------
    def on_fill(self, *, symbol: str, side: int, lots: float, price: float) -> bool:
        direction = "BUY" if side > 0 else "SELL"
        return self.notifier.send(AlertKind.FILL, f"{symbol} {direction} {lots:.2f} @ {price:.5f}")

    def on_regime(self, regime: Regime | str) -> bool:
        rv = regime.value if isinstance(regime, Regime) else str(regime)
        changed = self._last_regime is not None and rv != self._last_regime
        if changed:
            self.notifier.send(AlertKind.REGIME_FLIP, f"{self._last_regime} → {rv}")
        self._last_regime = rv
        return changed

    def update_equity(self, equity: float) -> bool:
        """Update peak equity; alert + return True on a drawdown-alert breach."""
        self._peak = max(self._peak, equity)
        dd = (self._peak - equity) / self._peak if self._peak > 0 else 0.0
        breached = dd >= self.config.drawdown_alert
        if breached:
            self.notifier.send(AlertKind.BREACH, f"drawdown {dd:.1%} (equity {equity:,.0f})")
        return breached

    def check_decay(self, returns: pd.Series, prob: np.ndarray, outcome: np.ndarray) -> DecayReport:
        """Evaluate decay; alert + flag retraining when edge/calibration degrades."""
        report = self.decay.evaluate(returns, prob, outcome)
        if report.retrain_due:
            self.notifier.send(AlertKind.DECAY, "; ".join(report.reasons()) or "model decay detected")
        return report

    # --- dashboard -----------------------------------------------------------
    def snapshot(
        self,
        *,
        equity: float,
        trades: pd.DataFrame | None = None,
        regime: Regime | str | None = None,
        open_positions: dict[str, float] | None = None,
    ) -> DashboardSnapshot:
        m = journal_metrics(
            trades, window=self.config.sharpe_window, periods_per_year=self.config.periods_per_year
        )
        peak = equity if self._peak == -math.inf else max(self._peak, equity)
        dd = (peak - equity) / peak if peak > 0 else 0.0
        rv = regime.value if isinstance(regime, Regime) else (regime or self._last_regime)
        retrain = (
            np.isfinite(m["rolling_sharpe"]) and m["rolling_sharpe"] < self.config.decay.min_rolling_sharpe
        ) or (
            np.isfinite(m["calibration_error"])
            and m["calibration_error"] > self.config.decay.max_calibration_error
        )
        return DashboardSnapshot(
            equity=equity, peak_equity=peak, drawdown=dd,
            rolling_sharpe=m["rolling_sharpe"], hit_rate=m["hit_rate"],
            calibration_error=m["calibration_error"], n_trades=m["n_trades"],
            total_pnl=m["total_pnl"], regime=rv, retrain_due=bool(retrain),
            open_positions=open_positions or {},
        )
