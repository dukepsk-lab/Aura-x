"""Layer 7 — Pre-trade execution guards.

Where a thin H4 edge is defended at the point of contact. These are pure,
testable predicates; the executor calls them before every order:

* spread guard   — skip if spread > threshold
* slippage guard — enforce max slippage between intended and fill price
* news guard      — skip inside high-impact news windows (NFP, ECB, BoE, FOMC)
"""

from __future__ import annotations

from datetime import datetime, timedelta


def spread_ok(spread_pips: float, max_spread_pips: float) -> bool:
    """True if the current spread is within budget."""
    return spread_pips <= max_spread_pips


def slippage_ok(intended_price: float, fill_price: float, pip_size: float, max_slippage_pips: float) -> bool:
    """True if |fill − intended| is within the max-slippage budget (in pips)."""
    slip_pips = abs(fill_price - intended_price) / pip_size
    return slip_pips <= max_slippage_pips


def in_news_window(
    now: datetime, events: list[datetime], blackout_minutes: int
) -> bool:
    """True if ``now`` is within ±``blackout_minutes`` of any high-impact event."""
    window = timedelta(minutes=blackout_minutes)
    return any(abs(now - ev) <= window for ev in events)
