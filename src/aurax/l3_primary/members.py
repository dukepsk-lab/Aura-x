"""Ensemble members for the primary SIDE model.

* :class:`LogisticSide`  — dependency-free multinomial logistic (numpy). The
  default/fallback member so the ensemble runs and tests anywhere.
* :class:`LightGBMSide`  — LightGBM gradient boosting (lazy import, ``[models]``).
* :class:`CatBoostSide`  — CatBoost gradient boosting (lazy import, ``[models]``).

A torch CNN / PatchTST / SSM member is a Roadmap-v2 addition: implement
:class:`~aurax.l3_primary.base.SideModel` and register it — no ensemble changes
needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import SideModel, align_proba, encode_labels, softmax


class LogisticSide(SideModel):
    """Multinomial logistic regression (softmax) via weighted gradient descent.

    No third-party ML dependency — keeps Layer 3 runnable with only numpy and
    gives the ensemble a sane, deterministic baseline member.
    """

    name = "logistic"

    def __init__(self, l2: float = 1e-3, lr: float = 0.5, n_iter: int = 400) -> None:
        self.l2 = l2
        self.lr = lr
        self.n_iter = n_iter

    def _design(self, X: pd.DataFrame) -> np.ndarray:
        z = (np.nan_to_num(X.to_numpy(dtype=float)) - self.mean_) / self.std_
        return np.c_[np.ones(len(z)), z]  # prepend bias

    def fit(self, X, y, sample_weight=None) -> LogisticSide:
        xv = np.nan_to_num(X.to_numpy(dtype=float))
        self.mean_ = xv.mean(axis=0)
        self.std_ = xv.std(axis=0) + 1e-9
        design = self._design(X)
        n, d = design.shape

        labels = encode_labels(y)
        self.classes_ = sorted(set(labels.tolist()))
        cls_idx = {c: i for i, c in enumerate(self.classes_)}
        c = len(self.classes_)
        target = np.zeros((n, c))
        target[np.arange(n), [cls_idx[v] for v in labels]] = 1.0

        w = (
            np.ones(n)
            if sample_weight is None
            else np.clip(np.asarray(sample_weight, dtype=float), 0, None)
        )
        w = w / w.sum()

        weights = np.zeros((c, d))
        reg_mask = np.ones(d)
        reg_mask[0] = 0.0  # don't regularise the bias
        for _ in range(self.n_iter):
            proba = softmax(design @ weights.T)
            grad = ((proba - target) * w[:, None]).T @ design + self.l2 * weights * reg_mask
            weights -= self.lr * grad
        self.weights_ = weights
        return self

    def predict_proba(self, X) -> pd.DataFrame:
        proba = softmax(self._design(X) @ self.weights_.T)
        return align_proba(proba, self.classes_, X.index)


class LightGBMSide(SideModel):
    """LightGBM multiclass member (lazy import; install ``[models]``)."""

    name = "lightgbm"

    # Objective/num_class are inferred by LGBMClassifier from y (a fold may hold
    # only 2 of the 3 classes), so they are deliberately not hard-coded here.
    _DEFAULTS = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "reg_lambda": 1.0,
        "verbosity": -1,
        "n_jobs": -1,
    }

    def __init__(self, **params) -> None:
        self.params = {**self._DEFAULTS, **params}

    def fit(self, X, y, sample_weight=None) -> LightGBMSide:
        from lightgbm import LGBMClassifier  # lazy: optional dependency

        encode_labels(y)  # validate
        self.model_ = LGBMClassifier(**self.params)
        self.model_.fit(X, np.asarray(y, dtype=int), sample_weight=sample_weight)
        self.classes_ = [int(c) for c in self.model_.classes_]
        return self

    def predict_proba(self, X) -> pd.DataFrame:
        return align_proba(self.model_.predict_proba(X), self.classes_, X.index)


class CatBoostSide(SideModel):
    """CatBoost multiclass member (lazy import; install ``[models]``)."""

    name = "catboost"

    # loss_function inferred from y (binary vs multiclass), as with LightGBM.
    _DEFAULTS = {
        "iterations": 300,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "verbose": False,
        "allow_writing_files": False,
    }

    def __init__(self, **params) -> None:
        self.params = {**self._DEFAULTS, **params}

    def fit(self, X, y, sample_weight=None) -> CatBoostSide:
        from catboost import CatBoostClassifier  # lazy: optional dependency

        encode_labels(y)
        self.model_ = CatBoostClassifier(**self.params)
        self.model_.fit(X, np.asarray(y, dtype=int), sample_weight=sample_weight)
        self.classes_ = [int(c) for c in self.model_.classes_]
        return self

    def predict_proba(self, X) -> pd.DataFrame:
        return align_proba(self.model_.predict_proba(X), self.classes_, X.index)


#: name → member class. Register new members (e.g. a CNN) here.
MEMBER_REGISTRY: dict[str, type[SideModel]] = {
    "logistic": LogisticSide,
    "lightgbm": LightGBMSide,
    "catboost": CatBoostSide,
}


def build_member(name: str, **params) -> SideModel:
    """Instantiate a registered member by name."""
    if name not in MEMBER_REGISTRY:
        raise KeyError(f"unknown member '{name}'; choices: {sorted(MEMBER_REGISTRY)}")
    return MEMBER_REGISTRY[name](**params)


def available_backends() -> list[str]:
    """Members whose backing library is importable in this environment."""
    import importlib.util

    ok = ["logistic"]
    if importlib.util.find_spec("lightgbm"):
        ok.append("lightgbm")
    if importlib.util.find_spec("catboost"):
        ok.append("catboost")
    return ok
