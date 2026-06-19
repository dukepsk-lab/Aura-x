"""Shared enumerations used across every layer.

Integer values are chosen so that ``Side``/``Label`` can be used directly in
arithmetic (``ret * side``) and stored compactly in TimescaleDB ``SMALLINT``
columns.
"""

from __future__ import annotations

from enum import Enum, IntEnum


class Timeframe(str, Enum):
    """Bar timeframes pulled by Layer 0."""

    M15 = "M15"   # execution context
    H4 = "H4"     # primary
    D1 = "D1"     # regime context

    @property
    def minutes(self) -> int:
        return {"M15": 15, "H4": 240, "D1": 1440}[self.value]


class Side(IntEnum):
    """Trade / signal direction. Usable directly in ``ret * side`` arithmetic."""

    SHORT = -1
    FLAT = 0
    LONG = 1


class Label(IntEnum):
    """Triple-barrier outcome label (direction of the barrier first touched)."""

    DOWN = -1     # lower (SL) barrier touched first
    NEUTRAL = 0   # vertical/timeout barrier touched first
    UP = 1        # upper (TP) barrier touched first


class BarrierTouch(str, Enum):
    """Which barrier was hit first in the triple-barrier method."""

    TP = "tp"             # upper / take-profit
    SL = "sl"             # lower / stop-loss
    VERTICAL = "vertical"  # timeout


class Regime(str, Enum):
    """Layer 2 regime classification."""

    TREND = "trend"
    RANGE = "range"
    SHOCK = "shock"   # high-volatility shock → system can stand down


class Session(str, Enum):
    """FX trading session (H4 bars behave very differently by session)."""

    ASIAN = "asian"
    LONDON = "london"
    NY = "ny"
    OFF = "off"


class Environment(str, Enum):
    DEV = "dev"
    STAGING = "staging"
    LIVE = "live"
