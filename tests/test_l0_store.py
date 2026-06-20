"""L0 DB-optional parquet stores (bars / features / labels).

These run anywhere with the ``[files]`` extra (pyarrow) — no DB, no MT5 — and
guarantee the parquet path is interface-compatible with the TimescaleDB
repositories and idempotent on rerun.
"""

from __future__ import annotations

import importlib.util
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from aurax.enums import Timeframe
from aurax.l0_data import (
    ParquetBarStore,
    ParquetFeatureStore,
    ParquetLabelStore,
    make_bar_store,
    make_feature_store,
    make_label_store,
)
from aurax.l0_data import store as store_mod

from .conftest import bars_from_close, h4_index

_NO_PARQUET = (
    importlib.util.find_spec("pyarrow") is None
    and importlib.util.find_spec("fastparquet") is None
)
_files = pytest.mark.skipif(_NO_PARQUET, reason="parquet engine (the `files` extra) not installed")

H4 = Timeframe.H4


def _bars(n: int = 120, start: str = "2020-01-01", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 1.10 + np.cumsum(rng.normal(0, 0.0015, size=n))
    df = bars_from_close(close, h4_index(n, start), seed=seed)
    df["spread"] = rng.uniform(0.5, 1.5, size=n)
    df.index.name = "ts"  # canonical bars are ts-indexed (matches rates_to_frame)
    return df


# --- bars --------------------------------------------------------------------
@_files
def test_bar_store_roundtrip_preserves_data(tmp_path):
    store = ParquetBarStore(tmp_path)
    bars = _bars()
    assert store.upsert_bars(bars, "EURUSD", H4) == len(bars)

    loaded = store.load_bars("EURUSD", H4)
    assert loaded.index.name == "ts"
    assert list(loaded.columns) == list(bars.columns)
    pd.testing.assert_frame_equal(loaded, bars, check_freq=False)


@_files
def test_bar_store_upsert_is_idempotent(tmp_path):
    store = ParquetBarStore(tmp_path)
    bars = _bars()
    store.upsert_bars(bars, "EURUSD", H4)
    store.upsert_bars(bars, "EURUSD", H4)  # rerun must not duplicate
    assert len(store.load_bars("EURUSD", H4)) == len(bars)


@_files
def test_bar_store_upsert_updates_on_conflict_and_appends(tmp_path):
    store = ParquetBarStore(tmp_path)
    bars = _bars(n=100, start="2020-01-01")
    store.upsert_bars(bars, "EURUSD", H4)

    # overlapping window with a changed close → keep last; plus new tail rows.
    overlap = bars.iloc[-10:].copy()
    overlap["close"] += 0.05
    later = _bars(n=20, start="2020-01-18", seed=5)  # disjoint tail
    store.upsert_bars(pd.concat([overlap, later]), "EURUSD", H4)

    loaded = store.load_bars("EURUSD", H4)
    assert len(loaded) == len(bars) + len(later)  # overlap updated, tail appended
    # the conflicting rows took the new (last-write-wins) close
    np.testing.assert_allclose(loaded.loc[overlap.index, "close"], overlap["close"])


@_files
def test_bar_store_load_respects_start_end_with_naive_bounds(tmp_path):
    store = ParquetBarStore(tmp_path)
    bars = _bars(n=200)
    store.upsert_bars(bars, "EURUSD", H4)

    # tz-naive bounds must still slice the tz-aware UTC index without raising.
    lo, hi = datetime(2020, 1, 5), datetime(2020, 1, 20)
    sliced = store.load_bars("EURUSD", H4, start=lo, end=hi)
    assert not sliced.empty
    assert sliced.index.min() >= pd.Timestamp(lo, tz="UTC")
    assert sliced.index.max() <= pd.Timestamp(hi, tz="UTC")
    assert len(sliced) < len(bars)


@_files
def test_bar_store_missing_and_empty_are_safe(tmp_path):
    store = ParquetBarStore(tmp_path)
    assert store.load_bars("EURUSD", H4).empty  # nothing written yet
    assert store.upsert_bars(pd.DataFrame(), "EURUSD", H4) == 0  # empty no-op
    assert not store._path("EURUSD", H4).exists()  # no stray file created


@_files
def test_bar_store_coverage_and_symbols(tmp_path):
    store = ParquetBarStore(tmp_path)
    assert store.coverage("EURUSD", H4) == (0, None, None)
    assert store.symbols() == []

    bars = _bars(n=50)
    store.upsert_bars(bars, "EURUSD", H4)
    store.upsert_bars(_bars(n=30, seed=2), "GBPUSD", H4)

    n, first, last = store.coverage("EURUSD", H4)
    assert n == 50
    assert first == bars.index.min()
    assert last == bars.index.max()
    assert store.symbols() == ["EURUSD", "GBPUSD"]


# --- features / labels -------------------------------------------------------
@_files
def test_feature_store_roundtrip(tmp_path):
    store = ParquetFeatureStore(tmp_path)
    idx = h4_index(40)
    matrix = pd.DataFrame(
        {"vol_atr_14": np.linspace(0.001, 0.002, 40), "tm_hurst": np.linspace(0.4, 0.6, 40)},
        index=idx,
    )
    assert store.upsert_features(matrix, "EURUSD", H4, "v1") == 40
    pd.testing.assert_frame_equal(store.load_features("EURUSD", H4, "v1"), matrix, check_freq=False)
    # a different feature_set is a separate file
    assert store.load_features("EURUSD", H4, "v2").empty


@_files
def test_label_store_roundtrip(tmp_path):
    store = ParquetLabelStore(tmp_path)
    idx = h4_index(12)
    labels = pd.DataFrame(
        {
            "t1": idx + pd.Timedelta("8h"),
            "label": np.array([1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0]),
            "ret": np.linspace(-0.01, 0.01, 12),
            "barrier": ["tp", "sl", "vertical"] * 4,
            "sample_weight": np.ones(12),
        },
        index=idx,
    )
    assert store.upsert_labels(labels, "EURUSD", H4, "tb_v1") == 12
    pd.testing.assert_frame_equal(store.load_labels("EURUSD", H4, "tb_v1"), labels, check_freq=False)


# --- factories ---------------------------------------------------------------
@_files
def test_factories_return_parquet_stores(tmp_path):
    assert isinstance(make_bar_store("parquet", tmp_path), ParquetBarStore)
    assert isinstance(make_feature_store("parquet", tmp_path), ParquetFeatureStore)
    assert isinstance(make_label_store("parquet", tmp_path), ParquetLabelStore)


def test_factories_reject_unknown_dest(tmp_path):
    for factory in (make_bar_store, make_feature_store, make_label_store):
        with pytest.raises(ValueError, match="unknown"):
            factory("sqlite", tmp_path)


def test_missing_parquet_engine_raises_clear_error(tmp_path, monkeypatch):
    # Simulate neither pyarrow nor fastparquet being importable.
    monkeypatch.setattr(store_mod.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match=r"\[files\]"):
        ParquetBarStore(tmp_path).upsert_bars(_bars(n=5), "EURUSD", H4)
