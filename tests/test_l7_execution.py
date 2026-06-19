"""L7 — guards, PaperBroker, and the guarded order path."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from aurax.enums import Regime, Side
from aurax.l6_risk import RiskConfig, RiskManager, SizingDecision, SizingReason, SizingRequest
from aurax.l7_execution import (
    ExecutionConfig,
    Executor,
    OrderRequest,
    OrderStatus,
    PaperBroker,
    in_news_window,
    slippage_ok,
    spread_ok,
)
from aurax.types import InstrumentSpec

_UTC = UTC
_NOW = datetime(2020, 1, 1, 12, 0, tzinfo=_UTC)
_EUR = InstrumentSpec("EURUSD", 0.0001, 5, 100_000, 0.01, 0.01, "EUR", "USD")
_GBP = InstrumentSpec("GBPUSD", 0.0001, 5, 100_000, 0.01, 0.01, "GBP", "USD")


def _decision(symbol="EURUSD", side=Side.LONG, lots=1.0, stop=0.0025, reason=SizingReason.OK):
    return SizingDecision(
        symbol, side, size_lots=float(int(side)) * lots, base_lots=2.0,
        confidence_mult=0.5, risk_fraction=0.0025, stop_distance=stop, reason=reason,
    )


def _executor(broker, journal=None, config=None):
    return Executor(broker, config or ExecutionConfig(), specs={"EURUSD": _EUR, "GBPUSD": _GBP}, journal=journal)


# --- guards ------------------------------------------------------------------
def test_guards():
    assert spread_ok(1.5, 2.0) and not spread_ok(2.5, 2.0)
    assert slippage_ok(1.1000, 1.10005, 0.0001, 1.0)        # 0.5 pip within budget
    assert not slippage_ok(1.1000, 1.1002, 0.0001, 1.0)     # 2.0 pip exceeds
    events = [datetime(2020, 1, 1, 12, 0, tzinfo=_UTC)]
    assert in_news_window(datetime(2020, 1, 1, 12, 20, tzinfo=_UTC), events, 30)
    assert not in_news_window(datetime(2020, 1, 1, 13, 30, tzinfo=_UTC), events, 30)


# --- PaperBroker -------------------------------------------------------------
def test_paper_broker_fills_with_adverse_slippage():
    b = PaperBroker(slippage_pips=0.5)
    b.set_quote("EURUSD", 1.0998, 1.1000)
    r = b.place_order(OrderRequest("id", "EURUSD", Side.LONG, 1.0, max_deviation_pips=1.0))
    assert r.status is OrderStatus.FILLED
    assert r.avg_price == pytest.approx(1.1000 + 0.5 * 0.0001)   # buy slips up
    assert b.positions["EURUSD"] == pytest.approx(1.0)


def test_paper_broker_rejects_excess_slippage_and_does_partials():
    b = PaperBroker(slippage_pips=2.0)
    b.set_quote("EURUSD", 1.0998, 1.1000)
    rej = b.place_order(OrderRequest("id", "EURUSD", Side.LONG, 1.0, max_deviation_pips=1.0))
    assert rej.status is OrderStatus.REJECTED

    b2 = PaperBroker(fill_ratio=0.5)
    b2.set_quote("EURUSD", 1.0998, 1.1000)
    part = b2.place_order(OrderRequest("id", "EURUSD", Side.LONG, 1.0))
    assert part.status is OrderStatus.PARTIAL and part.filled_lots == pytest.approx(0.5)


# --- Executor ----------------------------------------------------------------
def test_executor_fills_and_sets_atr_stops():
    b = PaperBroker()
    b.set_quote("EURUSD", 1.0998, 1.1000)
    r = _executor(b).execute(_decision(stop=0.0025), bid=1.0998, ask=1.1000, now=_NOW, bar_ts=_NOW)
    assert r.status is OrderStatus.FILLED and r.filled_lots == pytest.approx(1.0)
    order = b.orders[-1]
    assert order.sl == pytest.approx(1.1000 - 0.0025)             # SL = entry - stop
    assert order.tp == pytest.approx(1.1000 + 0.0025 * 1.5)       # TP at 1.5 R:R


def test_executor_spread_and_news_guards_skip():
    b = PaperBroker()
    b.set_quote("EURUSD", 1.0996, 1.1000)  # 4-pip spread
    ex = _executor(b, config=ExecutionConfig(max_spread_pips=2.0))
    wide = ex.execute(_decision(), bid=1.0996, ask=1.1000, now=_NOW, bar_ts=_NOW)
    assert wide.status is OrderStatus.SKIPPED and wide.reason == "spread"

    b.set_quote("EURUSD", 1.0999, 1.1000)  # 1-pip spread, ok
    news = ex.execute(
        _decision(), bid=1.0999, ask=1.1000, now=_NOW, bar_ts=_NOW, news_events=[_NOW]
    )
    assert news.status is OrderStatus.SKIPPED and news.reason == "news"


def test_executor_skips_flat_and_is_idempotent():
    b = PaperBroker()
    b.set_quote("EURUSD", 1.0999, 1.1000)
    ex = _executor(b)
    flat = ex.execute(_decision(side=Side.FLAT, lots=0.0, reason=SizingReason.FLAT),
                      bid=1.0999, ask=1.1000, now=_NOW, bar_ts=_NOW)
    assert flat.status is OrderStatus.SKIPPED and flat.reason == "no_position"

    first = ex.execute(_decision(), bid=1.0999, ask=1.1000, now=_NOW, bar_ts=_NOW)
    second = ex.execute(_decision(), bid=1.0999, ask=1.1000, now=_NOW, bar_ts=_NOW)
    assert first.status is OrderStatus.FILLED
    assert second.status is OrderStatus.SKIPPED and second.reason == "duplicate"


def test_executor_retries_on_requote():
    b = PaperBroker(requotes=2)
    b.set_quote("EURUSD", 1.0999, 1.1000)
    ex = _executor(b, config=ExecutionConfig(max_retries=3))
    r = ex.execute(_decision(), bid=1.0999, ask=1.1000, now=_NOW, bar_ts=_NOW)
    assert r.status is OrderStatus.FILLED  # survived two requotes


def test_executor_journals_fills():
    class FakeJournal:
        def __init__(self):
            self.records = []

        def record_open(self, rec):
            self.records.append(rec)

    b = PaperBroker()
    b.set_quote("EURUSD", 1.0999, 1.1000)
    j = FakeJournal()
    _executor(b, journal=j).execute(
        _decision(), bid=1.0999, ask=1.1000, now=_NOW, bar_ts=_NOW,
        meta_prob=0.7, regime=Regime.TREND,
    )
    assert len(j.records) == 1
    rec = j.records[0]
    assert rec["symbol"] == "EURUSD" and rec["side"] == 1 and rec["meta_prob"] == 0.7
    assert rec["regime"] == "trend"


# --- L6 → L7 integration -----------------------------------------------------
def test_risk_decision_flows_to_a_fill():
    rm = RiskManager(RiskConfig())
    decision = rm.evaluate(
        [SizingRequest("EURUSD", Side.LONG, 0.9, 0.0010, 1.10, _EUR)], equity=100_000
    )["EURUSD"]
    assert decision.approved

    b = PaperBroker()
    b.set_quote("EURUSD", 1.0999, 1.1000)
    r = _executor(b).execute(decision, bid=1.0999, ask=1.1000, now=_NOW, bar_ts=_NOW)
    assert r.is_fill and r.filled_lots == pytest.approx(abs(decision.size_lots))
