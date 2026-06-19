"""Broker abstraction for Layer 7.

The order path is written against a :class:`Broker` interface so it is testable
and paper-tradable without a live (Windows-only) MT5 terminal:

* :class:`PaperBroker` — deterministic in-memory broker with configurable spread,
  slippage, requotes and partial fills (for tests, paper trading, backtests);
* :class:`MT5Broker`   — thin ``MetaTrader5.order_send`` wrapper (lazy import).

All prices/quotes are in price units; volumes in lots; slippage budgets in pips.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from ..enums import Side


class OrderStatus(str, Enum):
    FILLED = "filled"
    PARTIAL = "partial"
    REJECTED = "rejected"
    REQUOTE = "requote"
    SKIPPED = "skipped"


@dataclass
class OrderRequest:
    """A broker-agnostic order (market by default)."""

    client_order_id: str         # idempotent — reruns of a bar must not double-fire
    symbol: str
    side: Side
    volume_lots: float
    pip_size: float = 0.0001
    price: float | None = None   # None → market
    sl: float | None = None
    tp: float | None = None
    max_deviation_pips: float = 1.5
    comment: str = "AuraX"
    magic: int = 990011


@dataclass
class OrderResult:
    client_order_id: str
    status: OrderStatus
    filled_lots: float = 0.0
    avg_price: float = 0.0
    broker_order_id: str | None = None
    reason: str = ""

    @property
    def is_fill(self) -> bool:
        return self.status in (OrderStatus.FILLED, OrderStatus.PARTIAL)


@runtime_checkable
class Broker(Protocol):
    def symbol_info(self, symbol: str) -> dict: ...
    def place_order(self, order: OrderRequest) -> OrderResult: ...


@dataclass
class PaperBroker:
    """Deterministic simulated broker.

    ``requotes`` returns ``REQUOTE`` that many times before filling (to exercise
    retry logic); ``fill_ratio`` < 1 produces partial fills; adverse ``slippage_pips``
    is applied and rejected if it exceeds the order's ``max_deviation_pips``.
    """

    slippage_pips: float = 0.0
    requotes: int = 0
    fill_ratio: float = 1.0
    _quotes: dict[str, tuple[float, float]] = field(default_factory=dict)
    _seq: int = 0
    positions: dict[str, float] = field(default_factory=dict)
    orders: list[OrderRequest] = field(default_factory=list)

    def set_quote(self, symbol: str, bid: float, ask: float) -> None:
        self._quotes[symbol] = (bid, ask)

    def symbol_info(self, symbol: str) -> dict:
        bid, ask = self._quotes[symbol]
        return {"bid": bid, "ask": ask, "spread": ask - bid}

    def place_order(self, order: OrderRequest) -> OrderResult:
        self.orders.append(order)
        if self.requotes > 0:  # transient requote → caller should retry
            self.requotes -= 1
            return OrderResult(order.client_order_id, OrderStatus.REQUOTE, reason="requote")

        bid, ask = self._quotes[order.symbol]
        base = order.price if order.price is not None else (ask if int(order.side) > 0 else bid)
        # adverse slippage: buys fill higher, sells lower
        fill = base + self.slippage_pips * order.pip_size * (1 if int(order.side) > 0 else -1)

        if self.slippage_pips > order.max_deviation_pips:
            return OrderResult(order.client_order_id, OrderStatus.REJECTED, reason="slippage>deviation")

        filled = order.volume_lots * self.fill_ratio
        self._seq += 1
        self.positions[order.symbol] = self.positions.get(order.symbol, 0.0) + int(order.side) * filled
        status = OrderStatus.FILLED if self.fill_ratio >= 1.0 else OrderStatus.PARTIAL
        return OrderResult(
            order.client_order_id, status, filled_lots=filled, avg_price=fill,
            broker_order_id=str(self._seq),
        )


class MT5Broker:
    """Live MT5 order routing (lazy import; Windows-only ``[mt5]`` extra).

    Maps :class:`OrderRequest` to ``MetaTrader5.order_send``. Full retcode
    handling / position management is the live-hardening task; the demo path and
    all tests use :class:`PaperBroker`.
    """

    def __init__(self) -> None:
        try:  # pragma: no cover - platform dependent
            import MetaTrader5 as mt5  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "MT5Broker requires MetaTrader5 (Windows). Install with: pip install -e '.[mt5]'"
            ) from exc
        self._mt5 = mt5

    def symbol_info(self, symbol: str) -> dict:  # pragma: no cover - needs terminal
        tick = self._mt5.symbol_info_tick(symbol)
        return {"bid": tick.bid, "ask": tick.ask, "spread": tick.ask - tick.bid}

    def place_order(self, order: OrderRequest) -> OrderResult:  # pragma: no cover - needs terminal
        mt5 = self._mt5
        otype = mt5.ORDER_TYPE_BUY if int(order.side) > 0 else mt5.ORDER_TYPE_SELL
        tick = mt5.symbol_info_tick(order.symbol)
        price = order.price or (tick.ask if int(order.side) > 0 else tick.bid)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": order.symbol,
            "volume": float(order.volume_lots),
            "type": otype,
            "price": price,
            "sl": order.sl or 0.0,
            "tp": order.tp or 0.0,
            "deviation": int(order.max_deviation_pips * 10),  # pips → points (5-digit)
            "magic": order.magic,
            "comment": order.comment,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        res = mt5.order_send(request)
        if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
            return OrderResult(order.client_order_id, OrderStatus.REJECTED, reason=str(getattr(res, "retcode", "none")))
        return OrderResult(
            order.client_order_id, OrderStatus.FILLED,
            filled_lots=res.volume, avg_price=res.price, broker_order_id=str(res.order),
        )
