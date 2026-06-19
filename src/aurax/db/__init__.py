"""TimescaleDB access layer (Layer 0 storage + feature/label/journal stores).

Requires the ``db`` optional dependency group::

    pip install -e ".[db]"
"""

from __future__ import annotations

from .engine import dispose_engine, get_engine
from .repository import BarRepository, FeatureRepository, LabelRepository, TradeRepository

__all__ = [
    "get_engine",
    "dispose_engine",
    "BarRepository",
    "FeatureRepository",
    "LabelRepository",
    "TradeRepository",
]
