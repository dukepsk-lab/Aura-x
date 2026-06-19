"""Layer 6 — Risk & Position Sizing.

Turns a gated signal (L3 side, L5 ``P(correct)``, L2 regime) plus an ATR and live
equity into a sized position, assembling the §6 rules into one hard risk gateway:

1. **Circuit breaker** — halt on a max-drawdown breach.
2. **Regime stand-down** — no size in a shock regime (the L2 safety gate).
3. **Confidence gating** — ``P < τ`` → no trade.
4. **ATR sizing** — ``risk_cash / (stop·ATR·contract)``: constant dollar risk,
   auto-shrinking when volatile.
5. **Confidence scaling** — × capped fractional-Kelly of ``P``.
6. **Correlation cap** — bound the *covariance* risk across EURUSD/GBPUSD so two
   correlated longs don't become one oversized bet.
7. **Lot rounding** — to broker volume constraints.

The pure formulas (1, 4–6) are standalone, tested functions; :class:`RiskManager`
wires them against live portfolio state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from ..enums import Regime, Side
from ..types import InstrumentSpec


# --- pure formulas -----------------------------------------------------------
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
    denom = (stop_multiplier * atr) * contract_size * quote_per_price
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


def apply_correlation_cap(exposures: dict[str, float], cap: float) -> dict[str, float]:
    """Scale signed exposures so gross ``Σ|exposure| ≤ cap`` (simple gross cap)."""
    gross = sum(abs(v) for v in exposures.values())
    if gross <= cap or gross == 0:
        return dict(exposures)
    factor = cap / gross
    return {k: v * factor for k, v in exposures.items()}


def correlation_matrix(n: int, rho: float) -> np.ndarray:
    """``n×n`` correlation matrix: 1 on the diagonal, ``rho`` off-diagonal."""
    c = np.full((n, n), rho, dtype=float)
    np.fill_diagonal(c, 1.0)
    return c


def aggregate_correlated_risk(signed_risks: np.ndarray, corr: np.ndarray) -> float:
    """Portfolio risk ``√(rᵀ C r)`` for signed per-leg risks ``r`` (the RL
    covariance penalty): correlated same-side legs add, opposite legs net off."""
    r = np.asarray(signed_risks, dtype=float)
    return float(np.sqrt(max(0.0, r @ corr @ r)))


# --- request / decision records ----------------------------------------------
class SizingReason(str, Enum):
    OK = "ok"
    FLAT = "flat"
    REGIME_STANDDOWN = "regime_standdown"
    META_GATE = "meta_gate"
    CIRCUIT_BREAKER = "circuit_breaker"
    CORR_CAPPED = "corr_capped"
    BELOW_MIN_LOT = "below_min_lot"


@dataclass
class SizingRequest:
    """A candidate trade arriving at L6 (already gated upstream by L5)."""

    symbol: str
    side: Side
    meta_prob: float
    atr: float
    price: float
    spec: InstrumentSpec
    regime: Regime | None = None


@dataclass
class SizingDecision:
    """L6's sized output for one instrument (feeds L7)."""

    symbol: str
    side: Side
    size_lots: float = 0.0       # signed, rounded to broker step
    base_lots: float = 0.0       # ATR base size, pre-confidence
    confidence_mult: float = 0.0
    risk_fraction: float = 0.0   # fraction of equity at risk
    stop_distance: float = 0.0   # stop_mult · ATR (price units) — for L7 stops
    reason: SizingReason = SizingReason.FLAT

    @property
    def approved(self) -> bool:
        return self.size_lots != 0.0


# --- config ------------------------------------------------------------------
@dataclass
class RiskConfig:
    risk_per_trade: float = 0.005
    atr_stop_mult: float = 2.5
    meta_threshold_tau: float = 0.55
    kelly_fraction: float = 0.5
    max_confidence_mult: float = 2.0
    correlation_cap: float = 0.015
    max_drawdown_breaker: float = 0.10
    assumed_correlation: float = 0.6   # EURUSD↔GBPUSD covariance for the cap
    account_ccy: str = "USD"
    specs: dict[str, InstrumentSpec] = field(default_factory=dict)

    @classmethod
    def from_params(cls, params: dict[str, Any], specs: dict[str, InstrumentSpec] | None = None) -> RiskConfig:
        r = params.get("risk", {})
        return cls(
            risk_per_trade=r.get("risk_per_trade", 0.005),
            atr_stop_mult=r.get("atr_stop_mult", 2.5),
            meta_threshold_tau=r.get("meta_threshold_tau", 0.55),
            kelly_fraction=r.get("kelly_fraction", 0.5),
            max_confidence_mult=r.get("max_confidence_mult", 2.0),
            correlation_cap=r.get("correlation_cap", 0.015),
            max_drawdown_breaker=r.get("max_drawdown_breaker", 0.10),
            assumed_correlation=r.get("assumed_correlation", 0.6),
            specs=specs or {},
        )


# --- manager -----------------------------------------------------------------
def round_to_lot(lots: float, spec: InstrumentSpec) -> float:
    """Round a (non-negative) lot magnitude down to the broker step, or 0 if
    below the minimum lot."""
    if lots <= 0 or spec.lot_step <= 0:
        return 0.0
    stepped = math.floor(lots / spec.lot_step) * spec.lot_step
    return round(stepped, 8) if stepped >= spec.min_lot else 0.0


class RiskManager:
    """Wire ATR sizing + confidence + correlation cap + circuit breaker."""

    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()
        self._peak_equity = -math.inf

    def circuit_breaker_tripped(self, equity: float) -> bool:
        """True once drawdown from peak breaches ``max_drawdown_breaker``."""
        self._peak_equity = max(self._peak_equity, equity)
        if self._peak_equity <= 0:
            return False
        return (self._peak_equity - equity) / self._peak_equity >= self.config.max_drawdown_breaker

    def _quote_per_price(self, spec: InstrumentSpec) -> float:
        # USD-quoted major on a USD account → 1.0; cross-currency conversion is
        # out of scope for v1's two USD majors.
        return 1.0

    def evaluate(
        self,
        requests: list[SizingRequest],
        equity: float,
        *,
        correlation: float | None = None,
    ) -> dict[str, SizingDecision]:
        """Size a set of candidate trades jointly (so the correlation cap binds
        across legs). Returns one :class:`SizingDecision` per request symbol."""
        cfg = self.config
        decisions = {
            r.symbol: SizingDecision(r.symbol, r.side, stop_distance=cfg.atr_stop_mult * r.atr)
            for r in requests
        }

        # 1) circuit breaker → halt everything.
        if self.circuit_breaker_tripped(equity) or equity <= 0:
            for d in decisions.values():
                d.reason = SizingReason.CIRCUIT_BREAKER
            return decisions

        # 2–5) per-leg gating + ATR + confidence sizing (provisional, pre-cap).
        active: list[SizingRequest] = []
        for r in requests:
            d = decisions[r.symbol]
            if int(r.side) == 0:
                d.reason = SizingReason.FLAT
                continue
            if r.regime == Regime.SHOCK:
                d.reason = SizingReason.REGIME_STANDDOWN
                continue
            conf = confidence_multiplier(
                r.meta_prob, tau=cfg.meta_threshold_tau,
                kelly_fraction=cfg.kelly_fraction, cap=cfg.max_confidence_mult,
            )
            if conf <= 0:
                d.reason = SizingReason.META_GATE
                continue
            qpp = self._quote_per_price(r.spec)
            base = position_size_atr(
                equity * cfg.risk_per_trade, r.atr,
                stop_multiplier=cfg.atr_stop_mult,
                contract_size=r.spec.contract_size, quote_per_price=qpp,
            )
            d.base_lots = base
            d.confidence_mult = conf
            d.size_lots = float(int(r.side)) * base * conf  # signed, pre-cap
            d.risk_fraction = cfg.risk_per_trade * conf
            d.reason = SizingReason.OK
            active.append(r)

        # 6) covariance-aware correlation cap across active legs.
        if len(active) >= 1:
            rho = cfg.assumed_correlation if correlation is None else correlation
            signed = np.array([decisions[r.symbol].risk_fraction * int(r.side) for r in active])
            agg = aggregate_correlated_risk(signed, correlation_matrix(len(active), rho))
            if agg > cfg.correlation_cap:
                scale = cfg.correlation_cap / agg
                for r in active:
                    d = decisions[r.symbol]
                    d.size_lots *= scale
                    d.risk_fraction *= scale
                    d.reason = SizingReason.CORR_CAPPED

        # 7) round to broker lot step; recompute realised risk fraction.
        for r in active:
            d = decisions[r.symbol]
            rounded = round_to_lot(abs(d.size_lots), r.spec)
            if rounded <= 0:
                d.size_lots, d.risk_fraction, d.reason = 0.0, 0.0, SizingReason.BELOW_MIN_LOT
                continue
            d.size_lots = float(int(r.side)) * rounded
            risk_cash = rounded * r.spec.contract_size * d.stop_distance * self._quote_per_price(r.spec)
            d.risk_fraction = risk_cash / equity
        return decisions
