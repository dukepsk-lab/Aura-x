"""Layer 7 — guarded MT5 order execution.

Turns an L6 :class:`~aurax.l6_risk.SizingDecision` into a broker order, defending
the thin H4 edge at the point of contact:

* **pre-trade guards** — spread / news / (post-fill) slippage;
* **idempotent order IDs** — one order per (symbol, bar), so reprocessing a bar
  never double-fires;
* **retry** on transient requotes, **partial-fill** aware;
* **ATR-scaled stops** — SL = entry ∓ ``stop_distance`` (from L6), TP at a
  reward:risk multiple;
* **journaling** — fills are written back to the trade journal (feeds L8 / CPCV).

Broker-agnostic (see :mod:`aurax.l7_execution.broker`): tests and paper trading
use :class:`PaperBroker`; live uses :class:`MT5Broker`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..enums import Regime
from ..l6_risk import SizingDecision
from ..logging import get_logger
from .broker import Broker, OrderRequest, OrderResult, OrderStatus
from .guards import in_news_window, spread_ok

log = get_logger(__name__)


@dataclass
class ExecutionConfig:
    max_spread_pips: float = 2.0
    max_slippage_pips: float = 1.5
    news_blackout_minutes: int = 30
    tp_rr: float = 1.5            # take-profit as reward:risk multiple of the stop
    max_retries: int = 3
    magic: int = 990011
    high_impact_events: list[str] = field(
        default_factory=lambda: ["NFP", "FOMC", "ECB", "BoE", "CPI"]
    )

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> ExecutionConfig:
        e = params.get("execution", {})
        return cls(
            max_spread_pips=e.get("max_spread_pips", 2.0),
            max_slippage_pips=e.get("max_slippage_pips", 1.5),
            news_blackout_minutes=e.get("news_blackout_minutes", 30),
            tp_rr=e.get("tp_rr", 1.5),
            max_retries=e.get("max_retries", 3),
            magic=e.get("magic", 990011),
            high_impact_events=e.get("high_impact_events", ["NFP", "FOMC", "ECB", "BoE", "CPI"]),
        )


class Executor:
    """Guarded order placement against a :class:`Broker`."""

    def __init__(
        self,
        broker: Broker,
        config: ExecutionConfig | None = None,
        *,
        specs: dict[str, Any] | None = None,
        journal: Any | None = None,
    ) -> None:
        self.broker = broker
        self.config = config or ExecutionConfig()
        self.specs = specs or {}
        self.journal = journal
        self._seen: set[str] = set()

    # --- helpers -------------------------------------------------------------
    def _pip(self, symbol: str) -> float:
        spec = self.specs.get(symbol)
        return getattr(spec, "pip_size", 0.0001)

    def _order_id(self, symbol: str, ts: datetime | str) -> str:
        key = ts.strftime("%Y%m%d%H%M") if isinstance(ts, datetime) else str(ts)
        return f"AX-{self.config.magic}-{symbol}-{key}"

    def pre_trade_ok(
        self, *, spread_pips: float, now: datetime, news_events: Iterable[datetime] = ()
    ) -> bool:
        """Spread + news guards (slippage is enforced at fill by the broker)."""
        if not spread_ok(spread_pips, self.config.max_spread_pips):
            return False
        return not in_news_window(now, list(news_events), self.config.news_blackout_minutes)

    # --- execution -----------------------------------------------------------
    def execute(
        self,
        decision: SizingDecision,
        *,
        bid: float,
        ask: float,
        now: datetime,
        bar_ts: datetime | None = None,
        news_events: Iterable[datetime] = (),
        meta_prob: float | None = None,
        regime: Regime | None = None,
    ) -> OrderResult:
        """Guard, route, and (on fill) journal a single sized decision."""
        symbol = decision.symbol
        coid = self._order_id(symbol, bar_ts or now)

        if not decision.approved or int(decision.side) == 0:
            return OrderResult(coid, OrderStatus.SKIPPED, reason="no_position")
        if coid in self._seen:
            return OrderResult(coid, OrderStatus.SKIPPED, reason="duplicate")

        pip = self._pip(symbol)
        spread_pips = (ask - bid) / pip
        # A guard skip is not a commitment, so it does NOT consume the idempotent
        # id — only an actual broker send (below) does. That way a bar whose
        # spread later tightens can still fire, while a fill is never repeated.
        if not self.pre_trade_ok(spread_pips=spread_pips, now=now, news_events=news_events):
            reason = "spread" if not spread_ok(spread_pips, self.config.max_spread_pips) else "news"
            return OrderResult(coid, OrderStatus.SKIPPED, reason=reason)

        s = int(decision.side)
        entry = ask if s > 0 else bid
        order = OrderRequest(
            client_order_id=coid,
            symbol=symbol,
            side=decision.side,
            volume_lots=abs(decision.size_lots),
            pip_size=pip,
            sl=entry - s * decision.stop_distance,
            tp=entry + s * decision.stop_distance * self.config.tp_rr,
            max_deviation_pips=self.config.max_slippage_pips,
            magic=self.config.magic,
        )
        result = self._send_with_retry(order)
        self._seen.add(coid)

        if result.is_fill:
            self._journal(decision, result, now, spread_pips, meta_prob, regime)
        else:
            log.info("order_not_filled", symbol=symbol, status=result.status.value, reason=result.reason)
        return result

    def execute_all(
        self,
        decisions: dict[str, SizingDecision],
        quotes: dict[str, tuple[float, float]],
        *,
        now: datetime,
        bar_ts: datetime | None = None,
        news_events: Iterable[datetime] = (),
    ) -> dict[str, OrderResult]:
        """Execute every decision for which a (bid, ask) quote is available."""
        out: dict[str, OrderResult] = {}
        for symbol, decision in decisions.items():
            if symbol not in quotes:
                continue
            bid, ask = quotes[symbol]
            out[symbol] = self.execute(
                decision, bid=bid, ask=ask, now=now, bar_ts=bar_ts, news_events=news_events
            )
        return out

    def _send_with_retry(self, order: OrderRequest) -> OrderResult:
        result = OrderResult(order.client_order_id, OrderStatus.REJECTED, reason="no_attempt")
        for _ in range(self.config.max_retries + 1):
            result = self.broker.place_order(order)
            if result.status is not OrderStatus.REQUOTE:
                break
        return result

    def _journal(self, decision, result, now, spread_pips, meta_prob, regime) -> None:
        if self.journal is None:
            return
        try:
            self.journal.record_open(
                {
                    "trade_id": result.client_order_id,
                    "symbol": decision.symbol,
                    "side": int(decision.side),
                    "open_ts": now,
                    "entry_price": result.avg_price,
                    "size_lots": result.filled_lots,
                    "meta_prob": meta_prob,
                    "regime": regime.value if isinstance(regime, Regime) else regime,
                    "spread_entry": spread_pips,
                    "meta": {"stop_distance": decision.stop_distance, "reason": decision.reason.value},
                }
            )
        except Exception as exc:  # noqa: BLE001 - journaling must never block a fill
            log.warning("journal_failed", trade_id=result.client_order_id, err=str(exc))
