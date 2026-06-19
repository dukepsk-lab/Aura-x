"""L6 — ATR sizing, confidence scaling, correlation cap, circuit breaker."""

from __future__ import annotations

import pytest

from aurax.l6_risk import (
    RiskConfig,
    RiskManager,
    apply_correlation_cap,
    confidence_multiplier,
    position_size_atr,
)


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
