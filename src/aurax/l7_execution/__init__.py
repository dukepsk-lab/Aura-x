"""Layer 7 — Execution (guarded MT5 orders).

* :class:`Executor` / :class:`ExecutionConfig`  guarded order placement
* :class:`Broker` / :class:`PaperBroker` / :class:`MT5Broker`  broker backends
* :class:`OrderRequest` / :class:`OrderResult` / :class:`OrderStatus`
* guards: :func:`spread_ok`, :func:`slippage_ok`, :func:`in_news_window`
"""

from __future__ import annotations

from .broker import (
    Broker,
    MT5Broker,
    OrderRequest,
    OrderResult,
    OrderStatus,
    PaperBroker,
)
from .executor import ExecutionConfig, Executor
from .guards import in_news_window, slippage_ok, spread_ok

__all__ = [
    "Executor",
    "ExecutionConfig",
    "Broker",
    "PaperBroker",
    "MT5Broker",
    "OrderRequest",
    "OrderResult",
    "OrderStatus",
    "spread_ok",
    "slippage_ok",
    "in_news_window",
]
