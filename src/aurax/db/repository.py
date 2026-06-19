"""Repositories — typed read/write access to the TimescaleDB stores.

All writes are **idempotent upserts** (``ON CONFLICT ... DO UPDATE``) so reruns
of ingestion / feature builds / labeling never duplicate rows. Bulk paths take
and return :class:`pandas.DataFrame` to stay close to the vectorised layers.
"""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
from sqlalchemy import Engine, text

from ..enums import Timeframe
from .engine import get_engine

# --- Layer 0: bars -----------------------------------------------------------
_UPSERT_BARS = text(
    """
    INSERT INTO market.bars
        (symbol, timeframe, ts, open, high, low, close, volume, spread)
    VALUES
        (:symbol, :timeframe, :ts, :open, :high, :low, :close, :volume, :spread)
    ON CONFLICT (symbol, timeframe, ts) DO UPDATE SET
        open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
        close = EXCLUDED.close, volume = EXCLUDED.volume, spread = EXCLUDED.spread
    """
)


class BarRepository:
    """Read/write OHLCV bars."""

    def __init__(self, engine: Engine | None = None) -> None:
        self.engine = engine or get_engine()

    def upsert_bars(self, df: pd.DataFrame, symbol: str, timeframe: Timeframe) -> int:
        """Upsert a bar frame indexed by timestamp with OHLCV(+spread) columns."""
        if df.empty:
            return 0
        rows = [
            {
                "symbol": symbol,
                "timeframe": timeframe.value,
                "ts": ts.to_pydatetime(),
                "open": float(r.open),
                "high": float(r.high),
                "low": float(r.low),
                "close": float(r.close),
                "volume": float(getattr(r, "volume", 0.0) or 0.0),
                "spread": (None if pd.isna(getattr(r, "spread", None)) else float(r.spread)),
            }
            for ts, r in df.iterrows()
        ]
        with self.engine.begin() as conn:
            conn.execute(_UPSERT_BARS, rows)
        return len(rows)

    def load_bars(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Load bars into a timestamp-indexed frame (ascending)."""
        clauses = ["symbol = :symbol", "timeframe = :timeframe"]
        params: dict[str, object] = {"symbol": symbol, "timeframe": timeframe.value}
        if start is not None:
            clauses.append("ts >= :start")
            params["start"] = start
        if end is not None:
            clauses.append("ts <= :end")
            params["end"] = end
        sql = (
            "SELECT ts, open, high, low, close, volume, spread "
            "FROM market.bars WHERE " + " AND ".join(clauses) + " ORDER BY ts ASC"
        )
        df = pd.read_sql(text(sql), self.engine, params=params, parse_dates=["ts"])
        return df.set_index("ts")


# --- Layer 1: features -------------------------------------------------------
_UPSERT_FEATURES = text(
    """
    INSERT INTO market.features (symbol, timeframe, ts, feature_set, features)
    VALUES (:symbol, :timeframe, :ts, :feature_set, CAST(:features AS JSONB))
    ON CONFLICT (symbol, timeframe, feature_set, ts) DO UPDATE SET
        features = EXCLUDED.features, computed_at = now()
    """
)


class FeatureRepository:
    """Read/write the L1 feature store (JSONB feature vectors)."""

    def __init__(self, engine: Engine | None = None) -> None:
        self.engine = engine or get_engine()

    def upsert_features(
        self, df: pd.DataFrame, symbol: str, timeframe: Timeframe, feature_set: str = "v1"
    ) -> int:
        """Upsert a feature frame; each row becomes one JSONB vector."""
        if df.empty:
            return 0
        rows = [
            {
                "symbol": symbol,
                "timeframe": timeframe.value,
                "ts": ts.to_pydatetime(),
                "feature_set": feature_set,
                "features": json.dumps(
                    {k: (None if pd.isna(v) else float(v)) for k, v in row.items()}
                ),
            }
            for ts, row in df.iterrows()
        ]
        with self.engine.begin() as conn:
            conn.execute(_UPSERT_FEATURES, rows)
        return len(rows)


# --- Layer 4: labels ---------------------------------------------------------
_UPSERT_LABELS = text(
    """
    INSERT INTO market.labels
        (symbol, timeframe, ts, label_set, t1, label, ret, barrier,
         tp_price, sl_price, sample_weight, side)
    VALUES
        (:symbol, :timeframe, :ts, :label_set, :t1, :label, :ret, :barrier,
         :tp_price, :sl_price, :sample_weight, :side)
    ON CONFLICT (symbol, timeframe, label_set, ts) DO UPDATE SET
        t1 = EXCLUDED.t1, label = EXCLUDED.label, ret = EXCLUDED.ret,
        barrier = EXCLUDED.barrier, tp_price = EXCLUDED.tp_price,
        sl_price = EXCLUDED.sl_price, sample_weight = EXCLUDED.sample_weight,
        side = EXCLUDED.side
    """
)


class LabelRepository:
    """Write the L4 label store (training-only)."""

    def __init__(self, engine: Engine | None = None) -> None:
        self.engine = engine or get_engine()

    def upsert_labels(
        self, df: pd.DataFrame, symbol: str, timeframe: Timeframe, label_set: str = "tb_v1"
    ) -> int:
        """Upsert a labels frame produced by :mod:`aurax.l4_labeling`."""
        if df.empty:
            return 0
        rows = [
            {
                "symbol": symbol,
                "timeframe": timeframe.value,
                "ts": ts.to_pydatetime(),
                "label_set": label_set,
                "t1": (None if pd.isna(r.t1) else r.t1.to_pydatetime()),
                "label": int(r.label),
                "ret": (None if pd.isna(r.ret) else float(r.ret)),
                "barrier": str(r.barrier),
                "tp_price": float(r.tp_price),
                "sl_price": float(r.sl_price),
                "sample_weight": float(getattr(r, "sample_weight", 1.0)),
                "side": (None if pd.isna(getattr(r, "side", None)) else int(r.side)),
            }
            for ts, r in df.iterrows()
        ]
        with self.engine.begin() as conn:
            conn.execute(_UPSERT_LABELS, rows)
        return len(rows)


# --- Layer 7/8: trade journal ------------------------------------------------
class TradeRepository:
    """Append/update the trade journal (feeds L8 monitoring + CPCV refresh)."""

    def __init__(self, engine: Engine | None = None) -> None:
        self.engine = engine or get_engine()

    def record_open(self, trade: dict[str, object]) -> None:
        sql = text(
            """
            INSERT INTO market.trades
                (trade_id, symbol, side, open_ts, entry_price, size_lots,
                 meta_prob, regime, spread_entry, status, meta)
            VALUES
                (:trade_id, :symbol, :side, :open_ts, :entry_price, :size_lots,
                 :meta_prob, :regime, :spread_entry, 'open', CAST(:meta AS JSONB))
            ON CONFLICT (trade_id) DO NOTHING
            """
        )
        params = dict(trade)
        params["meta"] = json.dumps(params.get("meta") or {})
        with self.engine.begin() as conn:
            conn.execute(sql, params)
