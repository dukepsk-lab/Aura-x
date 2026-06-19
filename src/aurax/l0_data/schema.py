"""Normalise raw MT5 payloads into Aura-X's canonical frames.

Kept terminal-free so it is fully unit-testable: every function accepts anything
:class:`pandas.DataFrame` can consume (a numpy structured array from MT5, or a
plain list of dicts in tests).

MT5 conventions handled here:

* bar ``time`` is the bar **open** time in epoch **seconds**, broker/UTC;
* ``tick_volume`` is renamed to ``volume`` (real_volume is usually 0 on FX);
* tick ``time_msc`` (epoch **milliseconds**) is preferred over ``time`` for
  sub-second ordering when present.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

# Canonical column order written to ``market.bars`` / ``market.ticks``.
BAR_COLUMNS = ["open", "high", "low", "close", "volume", "spread"]
TICK_COLUMNS = ["bid", "ask", "last", "volume"]


def rates_to_frame(rates: Any) -> pd.DataFrame:
    """Normalise MT5 rates → OHLCV(+spread) frame indexed by UTC ``ts``.

    The index ``ts`` is the bar **open** time (point-in-time correct: a feature
    computed at ``ts`` may only use bars with index ``<= ts``).
    """
    df = pd.DataFrame(rates)
    if df.empty:
        return pd.DataFrame(columns=BAR_COLUMNS, index=pd.DatetimeIndex([], name="ts"))

    df = df.rename(columns={"tick_volume": "volume"})
    df["ts"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("ts").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    for col in BAR_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[BAR_COLUMNS]


def ticks_to_frame(ticks: Any) -> pd.DataFrame:
    """Normalise MT5 ticks → bid/ask/last/volume frame indexed by UTC ``ts``.

    A ``spread`` column (ask − bid) is added for convenience; the DB also stores
    it as a generated column.
    """
    df = pd.DataFrame(ticks)
    if df.empty:
        return pd.DataFrame(columns=[*TICK_COLUMNS, "spread"], index=pd.DatetimeIndex([], name="ts"))

    if "time_msc" in df.columns:
        df["ts"] = pd.to_datetime(df["time_msc"], unit="ms", utc=True)
    else:
        df["ts"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("ts").sort_index()

    for col in TICK_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df["spread"] = df["ask"] - df["bid"]
    return df[[*TICK_COLUMNS, "spread"]]
