"""Layer 1 — Feature Engineering.

Four blocks, all strictly point-in-time (features at bar ``t`` use only bars
``<= t``):

* :class:`VolatilityBlock`   ATR (multi-lookback), Yang-Zhang, realized vol, ATR-pct
* :class:`TrendMemoryBlock`  Hurst exponent, Kaufman Efficiency Ratio, DFA
* :class:`CrossPairBlock`    EURUSD↔GBPUSD correlation, spread z-score, lead-lag
* :class:`SessionBlock`      Asian/London/NY dummies + session-relative volatility

Compose them with :class:`FeaturePipeline`. ``atr`` is re-exported because Layer 4
reuses it for ATR-scaled barriers.
"""

from __future__ import annotations

from .base import FeatureBlock
from .cross_pair import CrossPairBlock
from .pipeline import FeaturePipeline, build_feature_matrix
from .session import SessionBlock
from .trend_memory import TrendMemoryBlock, hurst_exponent, kaufman_efficiency_ratio
from .volatility import VolatilityBlock, atr, realized_volatility, yang_zhang

__all__ = [
    "FeatureBlock",
    "FeaturePipeline",
    "build_feature_matrix",
    "VolatilityBlock",
    "TrendMemoryBlock",
    "CrossPairBlock",
    "SessionBlock",
    "atr",
    "yang_zhang",
    "realized_volatility",
    "hurst_exponent",
    "kaufman_efficiency_ratio",
]
