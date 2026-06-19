"""Layer 7 — MT5 order execution (scaffold).

Places orders through MT5 with pre-trade guards (spread/news/slippage), partial
-fill handling, retry logic and **idempotent order IDs**. Shares the
:class:`aurax.l0_data.MT5Client` terminal connection.

Only the guarded decision path is sketched here; live order send/modify is the
v1 build target (and must run against a demo account first).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from ..types import Signal
from .guards import in_news_window, spread_ok


@dataclass
class ExecutionConfig:
    max_spread_pips: float = 2.0
    max_slippage_pips: float = 1.5
    news_blackout_minutes: int = 30
    high_impact_events: list[str] = field(
        default_factory=lambda: ["NFP", "FOMC", "ECB", "BoE", "CPI"]
    )


class Executor:
    """Guarded MT5 order placement."""

    def __init__(self, config: ExecutionConfig | None = None, client: object | None = None) -> None:
        self.config = config or ExecutionConfig()
        self._client = client  # aurax.l0_data.MT5Client (injected at wire-up)

    def pre_trade_ok(
        self, *, spread_pips: float, now: datetime, news_events: list[datetime]
    ) -> bool:
        """Run the guard predicates that don't need a live fill (implemented)."""
        cfg = self.config
        if not spread_ok(spread_pips, cfg.max_spread_pips):
            return False
        return not in_news_window(now, news_events, cfg.news_blackout_minutes)

    def execute(self, signal: Signal) -> object:
        """Send the (already approved + sized) order to MT5. TODO(v1)."""
        raise NotImplementedError("MT5 order send lands in Roadmap v1 (L7).")
