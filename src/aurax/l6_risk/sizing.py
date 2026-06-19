"""Layer 6 — Risk & Position Sizing.

The §6 sizing formulas are simple and fully specified, so they are implemented
here as pure, tested functions:

* :func:`position_size_atr`      constant-dollar-risk ATR sizing
* :func:`confidence_multiplier`  capped fractional-Kelly scaling on meta-prob
* :func:`apply_correlation_cap`  bound aggregate shared-currency exposure

The stateful :class:`RiskManager` (which wires these together with the
volatility-target and the drawdown circuit breaker, using live equity/regime)
is the v1 build target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def position_size_atr(
    risk_cash: float,
    atr: float,
    *,
    stop_multiplier: float = 2.5,
    contract_size: float = 100_000.0,
    quote_per_price: float = 1.0,
) -> float:
    """Constant-dollar-risk position size in lots.

    Stop distance ``D = stop_multiplier · ATR`` (price units). Risking
    ``risk_cash`` over that distance on ``contract_size`` units per lot gives::

        lots = risk_cash / (D · contract_size · quote_per_price)

    ``quote_per_price`` converts a 1.0 price move per unit into account currency
    (1.0 for USD-quoted majors on a USD account). Auto-shrinks as ATR rises.
    """
    if atr <= 0 or stop_multiplier <= 0:
        return 0.0
    stop_distance = stop_multiplier * atr
    denom = stop_distance * contract_size * quote_per_price
    return 0.0 if denom <= 0 else risk_cash / denom


def confidence_multiplier(
    p: float,
    *,
    tau: float = 0.55,
    kelly_fraction: float = 0.5,
    cap: float = 2.0,
) -> float:
    """Capped fractional-Kelly multiplier from the calibrated meta-probability.

    Returns 0 below the gate ``τ`` (no trade). At ``p == τ`` the multiplier is
    ``kelly_fraction`` (small size near the threshold); it scales up with
    confidence and is hard-capped at ``cap`` to avoid over-betting on
    miscalibration. For even-money bets full Kelly is ``2p − 1``.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be a probability in [0, 1]")
    if p < tau:
        return 0.0
    edge = 2.0 * p - 1.0
    ref = 2.0 * tau - 1.0
    if ref <= 0:  # τ ≤ 0.5 → fall back to linear confidence above τ
        scaled = kelly_fraction * edge / (1.0 - tau) if tau < 1.0 else kelly_fraction
    else:
        scaled = kelly_fraction * edge / ref
    return float(min(cap, max(0.0, scaled)))


def apply_correlation_cap(
    exposures: dict[str, float], cap: float
) -> dict[str, float]:
    """Scale signed shared-currency exposures so ``Σ|exposure| ≤ cap``.

    EURUSD and GBPUSD share USD risk, so two correlated longs must not become one
    oversized bet (the RL "covariance risk penalty" as a hard gateway). Exposures
    are signed notionals in shared-currency units; if the aggregate breaches the
    cap, every leg is scaled by the same factor (preserving relative intent).
    """
    gross = sum(abs(v) for v in exposures.values())
    if gross <= cap or gross == 0:
        return dict(exposures)
    factor = cap / gross
    return {k: v * factor for k, v in exposures.items()}


@dataclass
class RiskConfig:
    risk_per_trade: float = 0.005
    atr_stop_mult: float = 2.5
    meta_threshold_tau: float = 0.55
    kelly_fraction: float = 0.5
    max_confidence_mult: float = 2.0
    correlation_cap: float = 0.015
    max_drawdown_breaker: float = 0.10


class RiskManager:
    """Wire sizing + correlation cap + circuit breaker against live state."""

    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()
        self._peak_equity = -math.inf

    def circuit_breaker_tripped(self, equity: float) -> bool:
        """True once drawdown from peak breaches ``max_drawdown_breaker``."""
        self._peak_equity = max(self._peak_equity, equity)
        if self._peak_equity <= 0:
            return False
        drawdown = (self._peak_equity - equity) / self._peak_equity
        return drawdown >= self.config.max_drawdown_breaker

    def evaluate(self, *args, **kwargs):  # noqa: ANN001
        """End-to-end size decision (regime + meta + vol-target). TODO(v1)."""
        raise NotImplementedError("RiskManager.evaluate lands in Roadmap v1 (L6).")
