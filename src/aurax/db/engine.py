"""SQLAlchemy engine factory for TimescaleDB.

A single pooled engine is shared process-wide. ``sqlalchemy`` is an optional
dependency (the ``[db]`` extra); importing this module without it raises a clear
error rather than an obscure ``ModuleNotFoundError`` deep in a call stack.
"""

from __future__ import annotations

import functools

try:
    from sqlalchemy import Engine, create_engine
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise ModuleNotFoundError(
        "aurax.db requires SQLAlchemy. Install with:  pip install -e '.[db]'"
    ) from exc

from ..config import get_settings


@functools.lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Return the process-wide pooled engine for TimescaleDB."""
    settings = get_settings()
    return create_engine(
        settings.db.dsn,
        pool_size=5,
        max_overflow=5,
        pool_pre_ping=True,
        future=True,
    )


def dispose_engine() -> None:
    """Dispose the cached engine (call on shutdown / between test sessions)."""
    if get_engine.cache_info().currsize:
        get_engine().dispose()
        get_engine.cache_clear()
