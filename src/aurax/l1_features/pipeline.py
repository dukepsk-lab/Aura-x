"""FeaturePipeline — compose feature blocks into a single point-in-time matrix.

Build it explicitly from blocks, or from merged YAML params via
:meth:`FeaturePipeline.from_params`. Cross-pair features are wired by passing the
partner leg's close series (the pipeline injects it into a
:class:`~aurax.l1_features.cross_pair.CrossPairBlock`).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..enums import Session
from .base import FeatureBlock, ensure_ohlcv
from .cross_pair import CrossPairBlock
from .session import SessionBlock
from .trend_memory import TrendMemoryBlock
from .volatility import VolatilityBlock


class FeaturePipeline:
    """Run an ordered set of feature blocks and concatenate their outputs."""

    def __init__(self, blocks: list[FeatureBlock]) -> None:
        self.blocks = blocks

    def compute(self, bars: pd.DataFrame, *, dropna_warmup: bool = False) -> pd.DataFrame:
        """Return the concatenated feature matrix aligned to ``bars`` index.

        With ``dropna_warmup=True`` the leading rows where *every* feature is NaN
        (indicator warm-up) are dropped; partial-NaN rows are kept so downstream
        layers decide their own NaN policy.
        """
        ensure_ohlcv(bars)
        frames = [block.compute(bars) for block in self.blocks]
        matrix = pd.concat(frames, axis=1)
        if dropna_warmup:
            matrix = matrix.loc[matrix.notna().any(axis=1)]
        return matrix

    @classmethod
    def from_params(
        cls, params: dict[str, Any], *, partner_close: pd.Series | None = None
    ) -> FeaturePipeline:
        """Construct the standard v1 block set from merged YAML params."""
        blocks: list[FeatureBlock] = [
            VolatilityBlock(**params.get("volatility", {})),
            TrendMemoryBlock(**params.get("trend_memory", {})),
            _session_block(params.get("session", {})),
        ]
        if partner_close is not None:
            xp = params.get("cross_pair", {})
            blocks.append(
                CrossPairBlock(
                    partner_close=partner_close,
                    corr_window=xp.get("corr_window", 48),
                    ratio_zscore_window=xp.get("ratio_zscore_window", 96),
                    lead_lag_max_shift=xp.get("lead_lag_max_shift", 6),
                )
            )
        return cls(blocks)


def _session_block(cfg: dict[str, Any]) -> SessionBlock:
    """Translate the ``session`` config dict into a :class:`SessionBlock`."""
    windows = {}
    for sess in (Session.ASIAN, Session.LONDON, Session.NY):
        w = cfg.get(sess.value)
        if w:
            windows[sess] = (int(w["start"]), int(w["end"]))
    return SessionBlock(
        windows=windows or None,
        session_relative_vol_window=cfg.get("session_relative_vol_window", 120),
    )


def build_feature_matrix(
    bars: pd.DataFrame,
    params: dict[str, Any],
    *,
    partner_close: pd.Series | None = None,
    dropna_warmup: bool = False,
) -> pd.DataFrame:
    """One-shot convenience: build the standard pipeline and compute features."""
    pipeline = FeaturePipeline.from_params(params, partner_close=partner_close)
    return pipeline.compute(bars, dropna_warmup=dropna_warmup)
