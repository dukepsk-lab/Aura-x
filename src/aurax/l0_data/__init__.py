"""Layer 0 — Data Ingestion & Storage.

MT5 pull of OHLCV (H4 primary, D1 regime context, M15 execution context) plus
tick/spread history, normalised to a canonical schema and written to TimescaleDB
— the single source of truth for both training and live inference.

Public surface:

* :class:`MT5Client`        thin, testable wrapper over the MetaTrader5 terminal
* :func:`rates_to_frame`    normalise raw MT5 rates → canonical OHLCV frame
* :func:`ticks_to_frame`    normalise raw MT5 ticks → canonical tick frame
* :class:`Ingestor`         orchestrates pull → normalise → upsert
* :class:`ParquetBarStore` / :class:`ParquetFeatureStore` / :class:`ParquetLabelStore`
  DB-optional local stores (parquet) + :func:`make_bar_store` /
  :func:`make_feature_store` / :func:`make_label_store` factories
"""

from __future__ import annotations

from .ingestion import Ingestor
from .mt5_client import MT5Client, MT5NotAvailableError
from .schema import rates_to_frame, ticks_to_frame
from .store import (
    ParquetBarStore,
    ParquetFeatureStore,
    ParquetLabelStore,
    make_bar_store,
    make_feature_store,
    make_label_store,
)

__all__ = [
    "MT5Client",
    "MT5NotAvailableError",
    "Ingestor",
    "rates_to_frame",
    "ticks_to_frame",
    "ParquetBarStore",
    "ParquetFeatureStore",
    "ParquetLabelStore",
    "make_bar_store",
    "make_feature_store",
    "make_label_store",
]
