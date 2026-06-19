"""Meta-labeling orchestration — the training path's L5 step, end to end.

The central loop of the whole system: the primary (L3) is a noisy idea
generator; the meta-model (L5) decides *whether to act* on each idea. It must be
trained on the primary's **out-of-fold** predictions, so this module generates
those, forms side-aware meta-labels, builds meta-features, and fits the
calibrated TRUST model.

* :func:`train_meta_labeler` — research/training path: returns the fitted meta
  model + diagnostics (precision before vs after the gate).
* :class:`MetaGatedPrimary` — sklearn-compatible wrapper (primary + meta gate)
  that drops straight into ``aurax.validation.run_validation``: ``predict`` emits
  the primary side, zeroed to flat wherever the meta-gate rejects.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from ..validation.backtest import CostModel
from ..validation.cpcv import PurgedKFold
from .features import build_meta_features
from .meta_model import MetaConfig, MetaLabelModel

PrimaryFactory = Callable[[], Any]  # () -> model with predict_proba(X, regimes)/predict_side


def meta_labels_from_sides(side: pd.Series, y: pd.Series) -> pd.Series:
    """Meta-label = 1 iff the primary acted *and* was right (``side == direction``)."""
    s = side.astype(int)
    d = y.reindex(s.index).astype(int)
    return ((s != 0) & (s == d)).astype(int)


def _oof_primary(
    primary_factory: PrimaryFactory,
    X: pd.DataFrame,
    y: pd.Series,
    t1: pd.Series,
    *,
    cv: PurgedKFold,
    regimes: pd.Series | None,
    sample_weight: pd.Series | None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Out-of-fold primary side + class probabilities (no event sees its own model)."""
    proba_cols = ["p_down", "p_flat", "p_up"]
    oof_proba = pd.DataFrame(np.nan, index=X.index, columns=proba_cols)
    oof_side = pd.Series(np.nan, index=X.index, dtype=float)
    yv = y.astype(int)
    for tr, te in cv.split(t1):
        model = primary_factory()
        w = None if sample_weight is None else sample_weight.iloc[tr]
        model.fit(X.iloc[tr], yv.iloc[tr], sample_weight=w)
        reg_te = None if regimes is None else regimes.reindex(X.index[te])
        oof_proba.iloc[te] = model.predict_proba(X.iloc[te], reg_te).to_numpy()
        oof_side.iloc[te] = model.predict_side(X.iloc[te], reg_te).to_numpy()
    return oof_side, oof_proba


def train_meta_labeler(
    primary_factory: PrimaryFactory,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    t1: pd.Series,
    regimes: pd.Series | None = None,
    context: pd.DataFrame | None = None,
    sample_weight: pd.Series | None = None,
    ret: pd.Series | None = None,
    cost_model: CostModel | None = None,
    meta_config: MetaConfig | None = None,
    tune: bool = True,
) -> dict[str, Any]:
    """Fit the meta-labeler on the primary's OOF predictions; return diagnostics.

    Diagnostics include the primary's raw bet precision vs the meta-gated
    precision — the headline "does the trust layer add value?" number.
    """
    cfg = meta_config or MetaConfig()
    cv = PurgedKFold(n_splits=cfg.cv_folds)
    oof_side, oof_proba = _oof_primary(
        primary_factory, X, y, t1, cv=cv, regimes=regimes, sample_weight=sample_weight
    )
    meta_y = meta_labels_from_sides(oof_side, y)
    feats = build_meta_features(
        oof_proba, oof_side, regimes=regimes, context=context, meta_labels=meta_y
    )

    acted = oof_side.fillna(0) != 0
    meta = MetaLabelModel(cfg)
    meta.fit(
        feats[acted], meta_y[acted],
        sample_weight=None if sample_weight is None else sample_weight[acted],
        t1=t1[acted],
    )
    probs = meta.predict_proba(feats[acted])

    if tune:
        meta.tune_threshold(
            probs, meta_y[acted],
            ret=None if ret is None else ret[acted],
            side=oof_side[acted],
            cost_model=cost_model,
        )

    taken = probs >= meta.threshold_
    return {
        "meta_model": meta,
        "meta_features": feats,
        "meta_labels": meta_y,
        "oof_side": oof_side,
        "tau": meta.threshold_,
        "precision_raw": float(meta_y[acted].mean()),
        "precision_gated": float(meta_y[acted][taken.to_numpy()].mean()) if taken.any() else float("nan"),
        "trade_rate_raw": float(acted.mean()),
        "trade_rate_gated": float((acted & acted.index.isin(probs.index[taken.to_numpy()])).mean()),
    }


class MetaGatedPrimary:
    """Primary ensemble + meta gate, as one sklearn-compatible estimator.

    ``predict`` returns the primary's side, set to flat (0) wherever the meta-gate
    rejects — so feeding it to ``run_validation`` measures the meta layer's effect
    on cost-adjusted, baseline-relative performance directly.

    ``t1`` (full label spans) and optional ``regimes`` are injected at construction
    and sliced per fold via the frame's index, since the validation harness's
    ``fit(X, y, sample_weight)`` does not thread them through.
    """

    def __init__(
        self,
        primary_factory: PrimaryFactory,
        *,
        t1: pd.Series,
        regimes: pd.Series | None = None,
        context_cols: list[str] | None = None,
        meta_config: MetaConfig | None = None,
    ) -> None:
        self.primary_factory = primary_factory
        self.t1 = t1
        self.regimes = regimes
        self.context_cols = context_cols or []
        self.meta_config = meta_config or MetaConfig()

    def _context(self, X: pd.DataFrame) -> pd.DataFrame | None:
        cols = [c for c in self.context_cols if c in X.columns]
        return X[cols] if cols else None

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None = None) -> MetaGatedPrimary:
        t1 = self.t1.reindex(X.index)
        regimes = None if self.regimes is None else self.regimes.reindex(X.index)
        cv = PurgedKFold(n_splits=self.meta_config.cv_folds)

        oof_side, oof_proba = _oof_primary(
            self.primary_factory, X, y, t1, cv=cv, regimes=regimes, sample_weight=sample_weight
        )
        meta_y = meta_labels_from_sides(oof_side, y)
        # No recent-win-rate feature here: it needs realised outcomes that aren't
        # symmetrically available at predict time (avoids train/serve skew).
        feats = build_meta_features(oof_proba, oof_side, regimes=regimes, context=self._context(X))

        acted = oof_side.fillna(0) != 0
        self.meta_ = MetaLabelModel(self.meta_config)
        if acted.sum() >= self.meta_config.cv_folds * 2:
            self.meta_.fit(
                feats[acted], meta_y[acted],
                sample_weight=None if sample_weight is None else sample_weight[acted],
                t1=t1[acted],
            )
            self._fitted = True
        else:  # too few primary bets to learn a trust layer → pass-through
            self._fitted = False

        # Refit the primary on all data for inference.
        self.primary_ = self.primary_factory()
        self.primary_.fit(X, y.astype(int), sample_weight=sample_weight)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        regimes = None if self.regimes is None else self.regimes.reindex(X.index)
        side = self.primary_.predict_side(X, regimes)
        if not self._fitted:
            return side.to_numpy()
        proba = self.primary_.predict_proba(X, regimes)
        feats = build_meta_features(proba, side, regimes=regimes, context=self._context(X))
        passed = self.meta_.gate(feats, self.meta_.threshold_)
        return side.where(passed, 0).to_numpy()
