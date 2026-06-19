"""L6 — ATR sizing, confidence scaling, correlation cap, circuit breaker."""

from __future__ import annotations

import numpy as np
import pytest

from aurax.enums import Regime, Side
from aurax.l6_risk import (
    RiskConfig,
    RiskManager,
    SizingReason,
    SizingRequest,
    aggregate_correlated_risk,
    apply_correlation_cap,
    confidence_multiplier,
    correlation_matrix,
    position_size_atr,
    round_to_lot,
)
from aurax.types import InstrumentSpec

_EUR = InstrumentSpec("EURUSD", 0.0001, 5, 100_000, 0.01, 0.01, "EUR", "USD")
_GBP = InstrumentSpec("GBPUSD", 0.0001, 5, 100_000, 0.01, 0.01, "GBP", "USD")


def _req(spec, side, p, atr=0.0010, price=1.10, regime=None):
    return SizingRequest(spec.symbol, side, p, atr, price, spec, regime)


def test_position_size_constant_dollar_risk():
    # risk 500, ATR 0.0010, stop 2.5x → stop distance 0.0025; 100k contract.
    lots = position_size_atr(500.0, 0.0010, stop_multiplier=2.5, contract_size=100_000)
    assert lots == pytest.approx(2.0)
    # auto-shrinks as volatility rises
    assert position_size_atr(500.0, 0.0020, stop_multiplier=2.5) < lots
    assert position_size_atr(500.0, 0.0) == 0.0


def test_confidence_multiplier_gate_and_cap():
    assert confidence_multiplier(0.50, tau=0.55) == 0.0          # below gate → no trade
    assert confidence_multiplier(0.55, tau=0.55, kelly_fraction=0.5) == pytest.approx(0.5)
    assert confidence_multiplier(0.99, tau=0.55, kelly_fraction=0.5, cap=2.0) == 2.0  # capped
    with pytest.raises(ValueError):
        confidence_multiplier(1.5)


def test_correlation_cap_scales_when_breached():
    capped = apply_correlation_cap({"EURUSD": 0.01, "GBPUSD": 0.01}, cap=0.015)
    assert sum(abs(v) for v in capped.values()) == pytest.approx(0.015)
    assert capped["EURUSD"] == pytest.approx(0.0075)
    # under the cap → untouched
    safe = apply_correlation_cap({"EURUSD": 0.005, "GBPUSD": 0.005}, cap=0.015)
    assert safe == {"EURUSD": 0.005, "GBPUSD": 0.005}


def test_circuit_breaker_trips_on_drawdown():
    rm = RiskManager(RiskConfig(max_drawdown_breaker=0.10))
    assert rm.circuit_breaker_tripped(1000.0) is False  # sets peak
    assert rm.circuit_breaker_tripped(950.0) is False   # 5% DD
    assert rm.circuit_breaker_tripped(900.0) is True    # 10% DD breach


# --- covariance-aware correlation risk ---------------------------------------
def test_aggregate_correlated_risk_covariance():
    same = np.array([0.01, 0.01])
    opp = np.array([0.01, -0.01])
    c = correlation_matrix(2, 0.6)
    assert aggregate_correlated_risk(same, c) > aggregate_correlated_risk(opp, c)
    # ρ=0 → independent → root-sum-square
    c0 = correlation_matrix(2, 0.0)
    assert aggregate_correlated_risk(same, c0) == pytest.approx(np.hypot(0.01, 0.01))


def test_round_to_lot():
    assert round_to_lot(1.234, _EUR) == pytest.approx(1.23)   # down to step
    assert round_to_lot(0.005, _EUR) == 0.0                   # below min lot
    assert round_to_lot(0.0, _EUR) == 0.0


# --- RiskManager.evaluate ----------------------------------------------------
def test_evaluate_basic_sizing():
    rm = RiskManager(RiskConfig())
    d = rm.evaluate([_req(_EUR, Side.LONG, 0.55)], equity=100_000)["EURUSD"]
    # base = 500 / (2.5·0.0010·100k) = 2.0 lots; conf(0.55)=0.5 → 1.0 lot.
    assert d.reason == SizingReason.OK
    assert d.size_lots == pytest.approx(1.0)
    assert d.base_lots == pytest.approx(2.0)
    assert d.stop_distance == pytest.approx(0.0025)
    assert d.approved


def test_evaluate_gates_flat_shock_and_low_confidence():
    rm = RiskManager(RiskConfig())
    out = rm.evaluate(
        [
            _req(_EUR, Side.FLAT, 0.9),
            _req(_GBP, Side.LONG, 0.9, regime=Regime.SHOCK),
        ],
        equity=100_000,
    )
    assert out["EURUSD"].reason == SizingReason.FLAT and out["EURUSD"].size_lots == 0.0
    assert out["GBPUSD"].reason == SizingReason.REGIME_STANDDOWN and not out["GBPUSD"].approved

    low = rm.evaluate([_req(_EUR, Side.LONG, 0.50)], equity=100_000)["EURUSD"]
    assert low.reason == SizingReason.META_GATE and low.size_lots == 0.0


def test_evaluate_circuit_breaker_halts_all():
    rm = RiskManager(RiskConfig(max_drawdown_breaker=0.10))
    rm.evaluate([_req(_EUR, Side.LONG, 0.9)], equity=100_000)  # sets peak
    out = rm.evaluate([_req(_EUR, Side.LONG, 0.9)], equity=88_000)  # 12% DD
    assert out["EURUSD"].reason == SizingReason.CIRCUIT_BREAKER
    assert out["EURUSD"].size_lots == 0.0


def test_correlation_cap_binds_two_correlated_longs():
    rm = RiskManager(RiskConfig(assumed_correlation=0.6, correlation_cap=0.015))
    reqs = [_req(_EUR, Side.LONG, 1.0), _req(_GBP, Side.LONG, 1.0)]  # conf=2.0 each
    out = rm.evaluate(reqs, equity=100_000)
    assert all(d.reason == SizingReason.CORR_CAPPED for d in out.values())
    # aggregate covariance risk pulled back to the cap
    signed = np.array([out["EURUSD"].risk_fraction, out["GBPUSD"].risk_fraction])
    agg = aggregate_correlated_risk(signed, correlation_matrix(2, 0.6))
    assert agg == pytest.approx(0.015, rel=0.05)
    # each leg smaller than its uncapped 4.0 lots
    assert out["EURUSD"].size_lots < 4.0


def test_opposite_legs_net_off_and_are_not_capped():
    rm = RiskManager(RiskConfig(assumed_correlation=0.6, correlation_cap=0.015))
    out = rm.evaluate(
        [_req(_EUR, Side.LONG, 1.0), _req(_GBP, Side.SHORT, 1.0)], equity=100_000
    )
    # long+short on positively-correlated majors hedges USD → under the cap.
    assert out["EURUSD"].reason == SizingReason.OK
    assert out["EURUSD"].size_lots == pytest.approx(4.0)
