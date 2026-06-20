"""Layer 0 — DB-optional local stores (parquet).

These mirror the TimescaleDB repositories' interface so the whole training path
(L1 features → L4 labels → validation) can run on real, MT5-pulled bars with
**no database**: a Windows MT5 host writes parquet, those files move anywhere
(including a Linux box with no Docker/Timescale), and the rest of the pipeline
reads them exactly as it would read TimescaleDB.

Layout under ``root`` (default ``<project>/data``)::

    bars/{symbol}/{timeframe}.parquet               OHLCV(+spread), index = ts (UTC)
    features/{symbol}/{timeframe}/{feature_set}.parquet   wide feature matrix
    labels/{symbol}/{timeframe}/{label_set}.parquet       triple-barrier labels

Writes are **idempotent upserts** (merge on the ``ts`` index, keep last, sort)
so reruns never duplicate rows — the same guarantee as the DB ``ON CONFLICT``
paths. The :class:`~aurax.db.repository.BarRepository` /
:class:`~aurax.db.repository.FeatureRepository` /
:class:`~aurax.db.repository.LabelRepository` signatures are matched method for
method, so a parquet store drops straight into ``Ingestor`` and the L1/L4
scripts without adapters.

``pyarrow`` is an optional dependency (the ``[files]`` extra); it is required
only when a store actually reads/writes — constructing one is dependency-free.
"""

from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..config import PROJECT_ROOT
from ..enums import Timeframe
from ..logging import get_logger

log = get_logger(__name__)

DEFAULT_ROOT = PROJECT_ROOT / "data"


# --- parquet engine + shared upsert/load helpers -----------------------------
def _require_parquet_engine() -> None:
    """Raise a clear, actionable error if no parquet engine is importable."""
    if (
        importlib.util.find_spec("pyarrow") is None
        and importlib.util.find_spec("fastparquet") is None
    ):
        raise RuntimeError(
            "Parquet stores require a parquet engine (pyarrow). "
            "Install with:  pip install -e '.[files]'"
        )


def _coerce_bound(value: datetime, like: pd.DatetimeIndex) -> pd.Timestamp:
    """Coerce a start/end bound to a Timestamp comparable with ``like``.

    The canonical bar index is tz-aware UTC; align a (possibly tz-naive) bound to
    it so ``df[df.index >= bound]`` never raises on a tz mismatch.
    """
    ts = pd.Timestamp(value)
    tz_aware = like.tz is not None
    if tz_aware and ts.tzinfo is None:
        return ts.tz_localize("UTC")
    if not tz_aware and ts.tzinfo is not None:
        return ts.tz_convert("UTC").tz_localize(None)
    return ts


def _upsert_parquet(path: Path, df: pd.DataFrame) -> int:
    """Merge ``df`` into the parquet at ``path`` on its index (keep last), sort, write.

    Returns the number of incoming rows (0 for an empty frame, which is a no-op).
    """
    if df.empty:
        return 0
    _require_parquet_engine()
    path.parent.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([pd.read_parquet(path), df]) if path.exists() else df.copy()
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined.to_parquet(path)
    return len(df)


def _load_parquet(
    path: Path, start: datetime | None = None, end: datetime | None = None
) -> pd.DataFrame:
    """Load a parquet frame (ascending), optionally sliced to ``[start, end]``."""
    if not path.exists():
        return pd.DataFrame()
    _require_parquet_engine()
    df = pd.read_parquet(path).sort_index()
    if isinstance(df.index, pd.DatetimeIndex):
        if start is not None:
            df = df[df.index >= _coerce_bound(start, df.index)]
        if end is not None:
            df = df[df.index <= _coerce_bound(end, df.index)]
    return df


# --- bars --------------------------------------------------------------------
class ParquetBarStore:
    """DB-optional OHLCV bar store — interface-compatible with ``BarRepository``."""

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root is not None else DEFAULT_ROOT

    def _path(self, symbol: str, timeframe: Timeframe) -> Path:
        return self.root / "bars" / symbol / f"{timeframe.value}.parquet"

    def upsert_bars(self, df: pd.DataFrame, symbol: str, timeframe: Timeframe) -> int:
        path = self._path(symbol, timeframe)
        n = _upsert_parquet(path, df)
        if n:
            log.info(
                "parquet_upsert_bars",
                symbol=symbol,
                timeframe=timeframe.value,
                rows=n,
                path=str(path),
            )
        return n

    def load_bars(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        df = _load_parquet(self._path(symbol, timeframe), start, end)
        if not df.empty:
            df.index.name = "ts"
        return df

    def coverage(
        self, symbol: str, timeframe: Timeframe
    ) -> tuple[int, pd.Timestamp | None, pd.Timestamp | None]:
        """Return ``(n_rows, first_ts, last_ts)`` — used by the preflight check."""
        df = self.load_bars(symbol, timeframe)
        if df.empty:
            return (0, None, None)
        return (len(df), df.index.min(), df.index.max())

    def symbols(self) -> list[str]:
        """Symbols with at least one stored bar file."""
        base = self.root / "bars"
        if not base.exists():
            return []
        return sorted(p.name for p in base.iterdir() if p.is_dir())


# --- features ----------------------------------------------------------------
class ParquetFeatureStore:
    """DB-optional L1 feature store — interface-compatible with ``FeatureRepository``.

    Unlike the DB path (one JSONB blob per row), the wide feature matrix is stored
    directly — lossless and cheaper to read back for validation.
    """

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root is not None else DEFAULT_ROOT

    def _path(self, symbol: str, timeframe: Timeframe, feature_set: str) -> Path:
        return self.root / "features" / symbol / timeframe.value / f"{feature_set}.parquet"

    def upsert_features(
        self, df: pd.DataFrame, symbol: str, timeframe: Timeframe, feature_set: str = "v1"
    ) -> int:
        path = self._path(symbol, timeframe, feature_set)
        n = _upsert_parquet(path, df)
        if n:
            log.info("parquet_upsert_features", symbol=symbol, rows=n, path=str(path))
        return n

    def load_features(
        self,
        symbol: str,
        timeframe: Timeframe,
        feature_set: str = "v1",
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        return _load_parquet(self._path(symbol, timeframe, feature_set), start, end)


# --- labels ------------------------------------------------------------------
class ParquetLabelStore:
    """DB-optional L4 label store — interface-compatible with ``LabelRepository``."""

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root is not None else DEFAULT_ROOT

    def _path(self, symbol: str, timeframe: Timeframe, label_set: str) -> Path:
        return self.root / "labels" / symbol / timeframe.value / f"{label_set}.parquet"

    def upsert_labels(
        self, df: pd.DataFrame, symbol: str, timeframe: Timeframe, label_set: str = "tb_v1"
    ) -> int:
        path = self._path(symbol, timeframe, label_set)
        n = _upsert_parquet(path, df)
        if n:
            log.info("parquet_upsert_labels", symbol=symbol, rows=n, path=str(path))
        return n

    def load_labels(
        self,
        symbol: str,
        timeframe: Timeframe,
        label_set: str = "tb_v1",
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        return _load_parquet(self._path(symbol, timeframe, label_set), start, end)


# --- store factories (dest = "parquet" | "db") -------------------------------
def make_bar_store(dest: str = "db", root: Path | str | None = None) -> object:
    """Return a bar store: a ``ParquetBarStore`` or the TimescaleDB ``BarRepository``."""
    if dest == "parquet":
        return ParquetBarStore(root)
    if dest == "db":
        from ..db import BarRepository  # local import: optional [db] extra

        return BarRepository()
    raise ValueError(f"unknown bar store dest {dest!r} (expected 'parquet' or 'db')")


def make_feature_store(dest: str = "db", root: Path | str | None = None) -> object:
    if dest == "parquet":
        return ParquetFeatureStore(root)
    if dest == "db":
        from ..db import FeatureRepository

        return FeatureRepository()
    raise ValueError(f"unknown feature store dest {dest!r} (expected 'parquet' or 'db')")


def make_label_store(dest: str = "db", root: Path | str | None = None) -> object:
    if dest == "parquet":
        return ParquetLabelStore(root)
    if dest == "db":
        from ..db import LabelRepository

        return LabelRepository()
    raise ValueError(f"unknown label store dest {dest!r} (expected 'parquet' or 'db')")
