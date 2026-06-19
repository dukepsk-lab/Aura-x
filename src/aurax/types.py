"""Shared, lightweight data records that cross layer boundaries.

Heavy numeric work happens on :class:`pandas.DataFrame` objects (vectorised);
these dataclasses are for the records that travel *between* layers and into/out
of the database, where explicit fields beat positional tuples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .enums import BarrierTouch, Label, Regime, Side, Timeframe


@dataclass(slots=True, frozen=True)
class InstrumentSpec:
    """Static contract specification for a tradable symbol (from config)."""

    symbol: str
    pip_size: float
    digits: int
    contract_size: float
    min_lot: float
    lot_step: float
    base_ccy: str
    quote_ccy: str
    asset_class: str = "fx_major"


@dataclass(slots=True)
class Bar:
    """A single OHLCV bar (Layer 0)."""

    symbol: str
    timeframe: Timeframe
    ts: datetime           # bar OPEN time (UTC)
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    spread: float | None = None


@dataclass(slots=True)
class Tick:
    """A single quote tick (Layer 0 — feeds realistic cost modeling)."""

    symbol: str
    ts: datetime
    bid: float
    ask: float
    last: float | None = None
    volume: float | None = None

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass(slots=True)
class LabelRecord:
    """One triple-barrier labeled event (Layer 4 → label store).

    Mirrors the ``market.labels`` table. ``label`` is the realised barrier
    direction; ``side`` is populated only in meta-labeling mode.
    """

    symbol: str
    timeframe: Timeframe
    ts: datetime                 # event start (bar close)
    t1: datetime | None          # first barrier touch / vertical timeout
    label: Label
    ret: float
    barrier: BarrierTouch
    tp_price: float
    sl_price: float
    sample_weight: float = 1.0
    side: Side | None = None


@dataclass(slots=True)
class RegimeState:
    """Layer 2 output conditioning L3/L6."""

    regime: Regime
    probabilities: dict[Regime, float] = field(default_factory=dict)
    hurst: float | None = None
    ker: float | None = None
    atr_percentile: float | None = None


@dataclass(slots=True)
class Signal:
    """End-to-end decision record flowing L3 → L5 → L6 → L7."""

    symbol: str
    ts: datetime
    side: Side                     # L3 primary direction
    meta_prob: float | None = None  # L5 P(signal correct), calibrated
    regime: Regime | None = None
    atr: float | None = None
    size_lots: float | None = None  # L6 position size
    approved: bool = False          # passed meta gate + risk + execution guards
