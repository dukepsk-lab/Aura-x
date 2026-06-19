"""Ingestor — orchestrates pull → normalise → upsert for Layer 0.

Pulls OHLCV (and optionally tick/spread history) from MT5, normalises via
:mod:`aurax.l0_data.schema`, and writes idempotently to TimescaleDB. Tick pulls
are chunked because tick history is large.

The DB write path imports :class:`aurax.db.BarRepository` lazily so that the
pull/normalise logic stays importable without the ``[db]`` extra installed.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from ..enums import Timeframe
from ..logging import get_logger
from .mt5_client import MT5Client
from .schema import rates_to_frame, ticks_to_frame

log = get_logger(__name__)


class Ingestor:
    """Coordinate market-data ingestion into TimescaleDB."""

    def __init__(
        self,
        client: MT5Client | None = None,
        *,
        bar_repo: object | None = None,
        tick_chunk_days: int = 7,
    ) -> None:
        self.client = client or MT5Client()
        self._bar_repo = bar_repo
        self.tick_chunk_days = tick_chunk_days

    # --- lazy repo (keeps pull/normalise usable without the [db] extra) ------
    def _bars(self):
        if self._bar_repo is None:
            from ..db import BarRepository  # local import: optional dependency

            self._bar_repo = BarRepository()
        return self._bar_repo

    # --- bars ----------------------------------------------------------------
    def fetch_bars(
        self, symbol: str, timeframe: Timeframe, start: datetime, end: datetime
    ) -> pd.DataFrame:
        """Pull and normalise bars (no DB write) — handy for backfills/tests."""
        rates = self.client.copy_rates(symbol, timeframe, start, end)
        df = rates_to_frame(rates)
        log.info("fetched_bars", symbol=symbol, timeframe=timeframe.value, n=len(df))
        return df

    def ingest_bars(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
        *,
        store: bool = True,
    ) -> int:
        """Pull bars and (by default) upsert them. Returns row count."""
        df = self.fetch_bars(symbol, timeframe, start, end)
        if store and not df.empty:
            self._bars().upsert_bars(df, symbol, timeframe)
        return len(df)

    # --- ticks ---------------------------------------------------------------
    def ingest_ticks(
        self, symbol: str, start: datetime, end: datetime, *, store: bool = True
    ) -> int:
        """Pull tick/spread history in chunks and upsert. Returns row count.

        Tick storage is non-negotiable: realistic cost modeling depends on it.
        """
        total = 0
        chunk = timedelta(days=self.tick_chunk_days)
        cursor = start
        frames: list[pd.DataFrame] = []
        while cursor < end:
            chunk_end = min(cursor + chunk, end)
            raw = self.client.copy_ticks(symbol, cursor, chunk_end)
            df = ticks_to_frame(raw)
            if not df.empty:
                frames.append(df)
                total += len(df)
            cursor = chunk_end
        log.info("fetched_ticks", symbol=symbol, n=total)
        if store and frames:
            from ..db import get_engine  # local import: optional dependency

            combined = pd.concat(frames)
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.assign(symbol=symbol).to_sql(
                "ticks", get_engine(), schema="market", if_exists="append", index_label="ts"
            )
        return total

    # --- convenience ---------------------------------------------------------
    def ingest_universe(
        self,
        pairs: list[str],
        timeframes: list[Timeframe],
        start: datetime,
        end: datetime,
        *,
        with_ticks: bool = False,
    ) -> dict[str, int]:
        """Ingest every (pair × timeframe); optionally ticks per pair."""
        summary: dict[str, int] = {}
        with self.client:
            for symbol in pairs:
                for tf in timeframes:
                    summary[f"{symbol}:{tf.value}"] = self.ingest_bars(symbol, tf, start, end)
                if with_ticks:
                    summary[f"{symbol}:ticks"] = self.ingest_ticks(symbol, start, end)
        return summary
