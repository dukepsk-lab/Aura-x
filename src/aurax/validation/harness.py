"""Validation harness — the §5 gatekeeper, end to end.

Runs the full protocol on a model and its Layer-4 labeled events:

1. carve a final **never-touched holdout**;
2. **CPCV** across the development period (edge stable across folds?);
3. **walk-forward** forward in time (edge survives sequentially?);
4. evaluate the holdout, after costs, against **all three baselines**;
5. **Deflated Sharpe** discounting for the number of configurations tried;
6. a **Go/No-Go** decision with explicit reasons.

The estimator is duck-typed (``fit``/``predict``) so this works with LightGBM /
CatBoost / sklearn or a trivial stub — no hard ML dependency. :func:`oof_predict`
is the bridge the Layer-5 meta-model trains on.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from .backtest import CostModel, strategy_returns, turnover
from .baselines import baseline_sharpes
from .cpcv import CombinatorialPurgedCV, PurgedKFold
from .metrics import (
    deflated_sharpe_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from .walkforward import WalkForwardSplit, holdout_split


@runtime_checkable
class Estimator(Protocol):
    """Minimal sklearn-style estimator contract."""

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: Any = ...) -> Any: ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


EstimatorFactory = Callable[[], Estimator]
PredictFn = Callable[[Estimator, pd.DataFrame], pd.Series]


def _default_predict(est: Estimator, X: pd.DataFrame) -> pd.Series:
    return pd.Series(np.asarray(est.predict(X)), index=X.index)


def _fit(est: Estimator, X: pd.DataFrame, y: pd.Series, w: pd.Series | None) -> Estimator:
    """Fit, passing ``sample_weight`` only if the estimator accepts it."""
    if w is None:
        est.fit(X, y)
    else:
        try:
            est.fit(X, y, sample_weight=w)
        except TypeError:
            est.fit(X, y)
    return est


@dataclass
class ValidationConfig:
    """Validation protocol parameters (mirrors ``config/default.yaml``)."""

    cpcv_n_groups: int = 6
    cpcv_n_test_groups: int = 2
    embargo_pct: float = 0.01
    walk_forward_splits: int = 5
    walk_forward_mode: str = "anchored"
    holdout_pct: float = 0.2
    periods_per_year: int = 1512
    deflated_sharpe_min: float = 0.95
    n_trials: int = 1
    cost_model: CostModel = field(default_factory=CostModel)

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> ValidationConfig:
        v = params.get("validation", {})
        return cls(
            cpcv_n_groups=v.get("cpcv_n_groups", 6),
            cpcv_n_test_groups=v.get("cpcv_n_test_groups", 2),
            embargo_pct=v.get("embargo_pct", 0.01),
            walk_forward_splits=v.get("walk_forward_splits", 5),
            walk_forward_mode=v.get("walk_forward_mode", "anchored"),
            holdout_pct=v.get("holdout_pct", 0.2),
            periods_per_year=v.get("periods_per_year", 1512),
            deflated_sharpe_min=v.get("deflated_sharpe_min", 0.95),
            n_trials=v.get("n_trials", 1),
        )


@dataclass
class ValidationReport:
    """Outcome of the validation gate, with an explicit Go/No-Go."""

    cpcv_sharpe_mean: float
    cpcv_sharpe_std: float
    cpcv_paths: int
    walk_forward_sharpe: float
    holdout_sharpe: float
    holdout_sortino: float
    holdout_max_drawdown: float
    holdout_turnover: float
    baselines: dict[str, float]
    best_baseline: float
    deflated_sharpe: float
    n_trials: int
    passed: bool
    reasons: list[str]

    def summary(self) -> str:
        verdict = "✅ GO" if self.passed else "⛔ NO-GO"
        lines = [
            f"Validation verdict: {verdict}",
            f"  CPCV Sharpe   : {self.cpcv_sharpe_mean:+.3f} ± {self.cpcv_sharpe_std:.3f} "
            f"over {self.cpcv_paths} paths",
            f"  Walk-forward  : {self.walk_forward_sharpe:+.3f}",
            f"  Holdout Sharpe: {self.holdout_sharpe:+.3f} | Sortino {self.holdout_sortino:+.3f} "
            f"| maxDD {self.holdout_max_drawdown:.3f} | turnover {self.holdout_turnover:.2f}",
            "  Baselines     : "
            + ", ".join(f"{k}={v:+.3f}" for k, v in self.baselines.items())
            + f"  (best {self.best_baseline:+.3f})",
            f"  Deflated SR   : {self.deflated_sharpe:.3f}  (n_trials={self.n_trials})",
        ]
        if self.reasons:
            lines.append("  Reasons       : " + "; ".join(self.reasons))
        return "\n".join(lines)


def oof_predict(
    make_estimator: EstimatorFactory,
    X: pd.DataFrame,
    y: pd.Series,
    t1: pd.Series,
    *,
    cv: PurgedKFold | None = None,
    sample_weight: pd.Series | None = None,
    predict_fn: PredictFn = _default_predict,
) -> pd.Series:
    """Out-of-fold predictions via purged k-fold — the Layer-5 training input.

    Every sample is predicted by a model that never saw it (nor any label
    overlapping it), so the meta-model learns the primary's *real* reliability.
    """
    cv = cv or PurgedKFold(n_splits=5)
    preds = pd.Series(np.nan, index=X.index, dtype=float)
    for train_pos, test_pos in cv.split(t1):
        w = None if sample_weight is None else sample_weight.iloc[train_pos]
        est = _fit(make_estimator(), X.iloc[train_pos], y.iloc[train_pos], w)
        preds.iloc[test_pos] = predict_fn(est, X.iloc[test_pos]).to_numpy()
    return preds


def _fold_returns(
    make_estimator: EstimatorFactory,
    X: pd.DataFrame,
    y: pd.Series,
    ret: pd.Series,
    entry_price: pd.Series | None,
    train_pos: np.ndarray,
    test_pos: np.ndarray,
    sample_weight: pd.Series | None,
    predict_fn: PredictFn,
    cost_model: CostModel,
) -> pd.Series:
    w = None if sample_weight is None else sample_weight.iloc[train_pos]
    est = _fit(make_estimator(), X.iloc[train_pos], y.iloc[train_pos], w)
    side = predict_fn(est, X.iloc[test_pos])
    ep = None if entry_price is None else entry_price.iloc[test_pos]
    return strategy_returns(side, ret.iloc[test_pos], entry_price=ep, cost_model=cost_model)


def run_validation(
    make_estimator: EstimatorFactory,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    t1: pd.Series,
    ret: pd.Series,
    labels: pd.Series,
    close: pd.Series | None = None,
    entry_price: pd.Series | None = None,
    sample_weight: pd.Series | None = None,
    config: ValidationConfig | None = None,
    predict_fn: PredictFn = _default_predict,
) -> ValidationReport:
    """Run the full §5 protocol and return a Go/No-Go :class:`ValidationReport`.

    All inputs share the labeled-event index. ``ret``/``labels`` are the L4
    ``ret``/``label`` columns; ``entry_price`` defaults to ``close`` at each event.
    """
    cfg = config or ValidationConfig()
    cm, ppy = cfg.cost_model, cfg.periods_per_year
    if entry_price is None and close is not None:
        entry_price = close.reindex(X.index)

    # 1) never-touched holdout off the end ------------------------------------
    dev_pos, hold_pos = holdout_split(t1, cfg.holdout_pct)
    Xd, yd, t1d, retd = X.iloc[dev_pos], y.iloc[dev_pos], t1.iloc[dev_pos], ret.iloc[dev_pos]
    wd = None if sample_weight is None else sample_weight.iloc[dev_pos]
    epd = None if entry_price is None else entry_price.iloc[dev_pos]

    # 2) CPCV across the development period ------------------------------------
    cpcv = CombinatorialPurgedCV(cfg.cpcv_n_groups, cfg.cpcv_n_test_groups, cfg.embargo_pct)
    path_sr_ann, path_sr_raw = [], []
    for tr, te in cpcv.split(t1d):
        r = _fold_returns(make_estimator, Xd, yd, retd, epd, tr, te, wd, predict_fn, cm)
        path_sr_ann.append(sharpe_ratio(r.to_numpy(), ppy))
        path_sr_raw.append(sharpe_ratio(r.to_numpy()))  # per-event, for DSR
    cpcv_mean = float(np.nanmean(path_sr_ann)) if path_sr_ann else float("nan")
    cpcv_std = float(np.nanstd(path_sr_ann)) if path_sr_ann else float("nan")

    # 3) walk-forward forward in time -----------------------------------------
    wf = WalkForwardSplit(cfg.walk_forward_splits, cfg.walk_forward_mode, cfg.embargo_pct)
    wf_parts = [
        _fold_returns(make_estimator, Xd, yd, retd, epd, tr, te, wd, predict_fn, cm)
        for tr, te in wf.split(t1d)
    ]
    wf_returns = pd.concat(wf_parts).sort_index() if wf_parts else pd.Series(dtype=float)
    wf_sharpe = sharpe_ratio(wf_returns.to_numpy(), ppy)

    # 4) holdout: train on ALL dev, evaluate the unseen tail ------------------
    est = _fit(make_estimator(), Xd, yd, wd)
    hold_side = predict_fn(est, X.iloc[hold_pos])
    ep_hold = None if entry_price is None else entry_price.iloc[hold_pos]
    hold_r = strategy_returns(hold_side, ret.iloc[hold_pos], entry_price=ep_hold, cost_model=cm)
    hold_sharpe = sharpe_ratio(hold_r.to_numpy(), ppy)
    hold_sharpe_raw = sharpe_ratio(hold_r.to_numpy())

    # 5) baselines on the SAME holdout universe -------------------------------
    close_hold = None if close is None else close.reindex(X.index).iloc[hold_pos]
    baselines = baseline_sharpes(
        labels.iloc[hold_pos], ret.iloc[hold_pos], close_hold,
        cost_model=cm, periods_per_year=ppy,
    )
    best_baseline = max(baselines.values()) if baselines else 0.0

    # 6) deflated Sharpe (per-event units), discounting trials ----------------
    sr_std_raw = float(np.nanstd(path_sr_raw)) if path_sr_raw else 0.0
    dsr = deflated_sharpe_ratio(
        hold_sharpe_raw, n_obs=len(hold_r), n_trials=cfg.n_trials, sharpe_std=sr_std_raw
    )

    # 7) Go/No-Go --------------------------------------------------------------
    reasons: list[str] = []
    if not (cpcv_mean > 0 and cpcv_mean > best_baseline):
        reasons.append(f"CPCV Sharpe {cpcv_mean:+.3f} fails to beat best baseline {best_baseline:+.3f}")
    if not wf_sharpe > 0:
        reasons.append(f"walk-forward Sharpe {wf_sharpe:+.3f} not positive")
    if not (hold_sharpe > 0 and hold_sharpe > best_baseline):
        reasons.append(f"holdout Sharpe {hold_sharpe:+.3f} fails to beat best baseline {best_baseline:+.3f}")
    if not dsr >= cfg.deflated_sharpe_min:
        reasons.append(f"deflated Sharpe {dsr:.3f} < {cfg.deflated_sharpe_min:.2f}")

    return ValidationReport(
        cpcv_sharpe_mean=cpcv_mean,
        cpcv_sharpe_std=cpcv_std,
        cpcv_paths=len(path_sr_ann),
        walk_forward_sharpe=wf_sharpe,
        holdout_sharpe=hold_sharpe,
        holdout_sortino=sortino_ratio(hold_r.to_numpy(), ppy),
        holdout_max_drawdown=max_drawdown(hold_r.to_numpy()),
        holdout_turnover=turnover(hold_side.reindex(ret.iloc[hold_pos].index).fillna(0.0)),
        baselines=baselines,
        best_baseline=best_baseline,
        deflated_sharpe=dsr,
        n_trials=cfg.n_trials,
        passed=not reasons,
        reasons=reasons,
    )
