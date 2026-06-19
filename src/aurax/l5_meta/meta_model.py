"""Layer 5 — Meta-Label Model (SIZE / TRUST).

The "self-trust" layer: predicts **P(primary signal is correct)** from the
primary's confidence plus regime / volatility / recent-performance features, and
gates trades at ``P ≥ τ``. Tuned for **high precision** — filtering false
positives is what protects the thin H4 cost budget.

* :class:`LogisticMeta`   — dependency-free binary logistic base learner.
* :class:`MetaLabelModel` — base learner + **cross-validated calibration** so the
  probability is a usable sizing input (never calibrated in-sample).

The meta-model must be trained on the primary's **out-of-fold** predictions (see
:mod:`aurax.l5_meta.pipeline`), never in-sample, or the trust layer learns nothing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..validation.backtest import CostModel, strategy_returns
from ..validation.cpcv import PurgedKFold
from ..validation.metrics import sharpe_ratio
from .calibration import make_calibrator


class LogisticMeta:
    """Binary logistic regression (numpy) — the default meta base learner."""

    def __init__(self, l2: float = 1e-2, lr: float = 0.3, n_iter: int = 500) -> None:
        self.l2 = l2
        self.lr = lr
        self.n_iter = n_iter

    def _design(self, X: pd.DataFrame) -> np.ndarray:
        z = (np.nan_to_num(X.to_numpy(dtype=float)) - self.mean_) / self.std_
        return np.c_[np.ones(len(z)), z]

    def fit(self, X, y, sample_weight=None) -> LogisticMeta:
        xv = np.nan_to_num(X.to_numpy(dtype=float))
        self.mean_, self.std_ = xv.mean(axis=0), xv.std(axis=0) + 1e-9
        design = self._design(X)
        target = np.asarray(y, dtype=float)
        w = (
            np.ones(len(target))
            if sample_weight is None
            else np.clip(np.asarray(sample_weight, dtype=float), 0, None)
        )
        w = w / w.sum()
        coef = np.zeros(design.shape[1])
        reg = np.ones(design.shape[1])
        reg[0] = 0.0  # don't regularise bias
        for _ in range(self.n_iter):
            p = 1.0 / (1.0 + np.exp(-np.clip(design @ coef, -30, 30)))
            grad = design.T @ ((p - target) * w) + self.l2 * coef * reg
            coef -= self.lr * grad
        self.coef_ = coef
        return self

    def predict_proba(self, X) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(self._design(X) @ self.coef_, -30, 30)))


def _proba1(estimator: Any, X: pd.DataFrame) -> np.ndarray:
    """P(class 1) from either a 1-D ``predict_proba`` or an sklearn 2-column one."""
    p = np.asarray(estimator.predict_proba(X))
    return p[:, 1] if p.ndim == 2 else p


def _default_meta_base() -> LogisticMeta:
    """Default base learner factory (a module-level fn so the model stays picklable)."""
    return LogisticMeta()


@dataclass
class MetaConfig:
    base: str = "logistic"            # base learner ('logistic' or injected)
    calibration: str = "isotonic"     # 'isotonic' | 'sigmoid' | 'none'
    threshold_tau: float = 0.55       # trade iff P(correct) >= tau
    cv_folds: int = 5

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> MetaConfig:
        m = params.get("meta", {})
        return cls(
            base=m.get("base", "logistic"),
            calibration=m.get("calibration", "isotonic"),
            threshold_tau=m.get("threshold_tau", 0.55),
            cv_folds=m.get("cv_folds", 5),
        )


class MetaLabelModel:
    """Calibrated binary meta-learner predicting P(primary signal correct)."""

    def __init__(
        self,
        config: MetaConfig | None = None,
        base_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.config = config or MetaConfig()
        self._base_factory = base_factory or _default_meta_base
        self.threshold_ = self.config.threshold_tau

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: pd.Series | None = None,
        t1: pd.Series | None = None,
    ) -> MetaLabelModel:
        """Fit base + calibrator. Calibration is cross-validated (purged if ``t1``
        is given) so the probability map is never learned in-sample."""
        yv = np.asarray(y, dtype=float)
        oof = np.full(len(X), np.nan)
        for tr, te in self._splits(X, t1):
            w = None if sample_weight is None else sample_weight.iloc[tr]
            base = self._base_factory()
            base.fit(X.iloc[tr], yv[tr], sample_weight=w)
            oof[te] = _proba1(base, X.iloc[te])

        mask = ~np.isnan(oof)
        cal_w = None if sample_weight is None else sample_weight.to_numpy()[mask]
        self.calibrator_ = make_calibrator(self.config.calibration).fit(
            oof[mask], yv[mask], cal_w
        )
        # Refit the base on all data for inference.
        self.base_ = self._base_factory()
        self.base_.fit(X, yv, sample_weight=sample_weight)
        return self

    def _splits(self, X: pd.DataFrame, t1: pd.Series | None):
        if t1 is not None:
            yield from PurgedKFold(n_splits=self.config.cv_folds).split(t1)
        else:  # contiguous fallback when label spans are unavailable
            folds = np.array_split(np.arange(len(X)), self.config.cv_folds)
            for i in range(len(folds)):
                test = folds[i]
                train = np.concatenate([folds[j] for j in range(len(folds)) if j != i])
                yield train, test

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """Calibrated P(primary signal correct)."""
        raw = _proba1(self.base_, X)
        return pd.Series(self.calibrator_.predict(raw), index=X.index, name="p_correct")

    def gate(self, X: pd.DataFrame, tau: float | None = None) -> pd.Series:
        """Boolean trade gate: ``P(correct) ≥ τ``."""
        tau = self.threshold_ if tau is None else tau
        return self.predict_proba(X) >= tau

    def tune_threshold(
        self,
        probs: pd.Series,
        meta_labels: pd.Series,
        *,
        ret: pd.Series | None = None,
        side: pd.Series | None = None,
        cost_model: CostModel | None = None,
        grid: np.ndarray | None = None,
        min_trades: int = 20,
    ) -> float:
        """Pick τ maximising a cost-adjusted objective (§6: tuned on cost-adjusted CV).

        With ``ret``+``side`` → maximise net-of-cost Sharpe of the gated trades;
        otherwise maximise precision (subject to ``min_trades``). Stores and
        returns the chosen τ.
        """
        grid = np.round(np.arange(0.40, 0.86, 0.01), 2) if grid is None else grid
        best_tau, best_obj = self.config.threshold_tau, -np.inf
        labels = meta_labels.to_numpy(dtype=float)
        for tau in grid:
            take = probs.to_numpy() >= tau
            if take.sum() < min_trades:
                continue
            if ret is not None and side is not None:
                gated_side = side.where(pd.Series(take, index=side.index), 0.0)
                r = strategy_returns(gated_side, ret, cost_model=cost_model)
                obj = sharpe_ratio(r.to_numpy())
            else:
                obj = labels[take].mean()  # precision
            if np.isfinite(obj) and obj > best_obj:
                best_obj, best_tau = obj, float(tau)
        self.threshold_ = best_tau
        return best_tau
