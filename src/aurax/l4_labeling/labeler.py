"""Labeler — orchestrates Layer 4 (triple-barrier + uniqueness + trend-scanning).

Produces a label frame ready for ``market.labels`` (via
:class:`aurax.db.LabelRepository`) or for direct training. ATR for barrier
scaling is reused from Layer 1 (:func:`aurax.l1_features.atr`) so labels and
features share one volatility definition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..l1_features import atr
from .trend_scanning import trend_scanning_labels
from .triple_barrier import triple_barrier_labels
from .uniqueness import sample_weights


@dataclass
class LabelConfig:
    """Layer 4 parameters (mirrors ``config/labeling.yaml``)."""

    # triple_barrier
    atr_lookback: int = 14
    tp_atr_mult: float = 2.0
    sl_atr_mult: float = 2.0
    vertical_bars: int = 12
    min_return: float = 0.0
    side_aware: bool = False
    tie_break: str = "sl"
    # uniqueness
    uniqueness_enabled: bool = True
    apply_time_decay: bool = True
    time_decay_last_weight: float = 0.5
    by_return: bool = False
    # trend_scanning
    trend_scanning_enabled: bool = False
    ts_min_horizon: int = 5
    ts_max_horizon: int = 20
    ts_t_value_threshold: float = 2.0

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> LabelConfig:
        tb = params.get("triple_barrier", {})
        uq = params.get("uniqueness", {})
        ts = params.get("trend_scanning", {})
        return cls(
            atr_lookback=tb.get("atr_lookback", 14),
            tp_atr_mult=tb.get("tp_atr_mult", 2.0),
            sl_atr_mult=tb.get("sl_atr_mult", 2.0),
            vertical_bars=tb.get("vertical_bars", 12),
            min_return=tb.get("min_return", 0.0),
            side_aware=tb.get("side_aware", False),
            uniqueness_enabled=uq.get("enabled", True),
            apply_time_decay=uq.get("apply_time_decay", True),
            time_decay_last_weight=uq.get("time_decay_last_weight", 0.5),
            trend_scanning_enabled=ts.get("enabled", False),
            ts_min_horizon=ts.get("min_horizon", 5),
            ts_max_horizon=ts.get("max_horizon", 20),
            ts_t_value_threshold=ts.get("t_value_threshold", 2.0),
        )


@dataclass
class Labeler:
    """Generate triple-barrier labels (+ uniqueness weights) for a bar frame."""

    config: LabelConfig = field(default_factory=LabelConfig)

    def make_labels(
        self,
        bars: pd.DataFrame,
        *,
        events: pd.DatetimeIndex | None = None,
        side: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Triple-barrier label every event and attach sample weights.

        ``side`` (primary L3 predictions) switches on meta-labeling mode; if
        omitted and ``config.side_aware`` is False, labels are direction labels
        for training the primary model.
        """
        cfg = self.config
        barrier_atr = atr(bars, cfg.atr_lookback)

        labels = triple_barrier_labels(
            bars,
            barrier_atr,
            tp_mult=cfg.tp_atr_mult,
            sl_mult=cfg.sl_atr_mult,
            vertical_bars=cfg.vertical_bars,
            events=events,
            side=side,
            min_ret=cfg.min_return,
            tie_break=cfg.tie_break,
        )
        if labels.empty:
            labels["sample_weight"] = pd.Series(dtype=float)
            return labels

        if cfg.uniqueness_enabled:
            labels["sample_weight"] = sample_weights(
                labels["t1"],
                bars.index,
                close=bars["close"],
                by_return=cfg.by_return,
                apply_time_decay=cfg.apply_time_decay,
                time_decay_last_weight=cfg.time_decay_last_weight,
            ).reindex(labels.index)
        else:
            labels["sample_weight"] = 1.0
        return labels

    def make_trend_labels(
        self, bars: pd.DataFrame, *, events: pd.DatetimeIndex | None = None
    ) -> pd.DataFrame:
        """Optional trend-scanning labels for the trend-regime sub-model."""
        cfg = self.config
        return trend_scanning_labels(
            bars["close"],
            min_horizon=cfg.ts_min_horizon,
            max_horizon=cfg.ts_max_horizon,
            t_value_threshold=cfg.ts_t_value_threshold,
            events=events,
        )
