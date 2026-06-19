"""Layer 7 — Execution (MT5 orders + spread/news/slippage guards)."""

from __future__ import annotations

from .executor import ExecutionConfig, Executor
from .guards import in_news_window, slippage_ok, spread_ok

__all__ = ["Executor", "ExecutionConfig", "spread_ok", "slippage_ok", "in_news_window"]
