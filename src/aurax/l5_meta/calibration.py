"""Probability calibration for the meta-model.

Calibration (Platt / isotonic) is essential so the meta-model's output is a
usable *sizing input*, not just a classifier score: at ``P=0.7`` the signal
should be correct ~70% of the time. Both calibrators are hand-rolled (numpy
only) so Layer 5 runs without ``scikit-learn``.

* :class:`IsotonicCalibrator` — monotonic, non-parametric (pool-adjacent-violators).
* :class:`SigmoidCalibrator`  — Platt scaling, ``P = σ(a·s + b)``.
* :class:`IdentityCalibrator` — pass-through (``calibration="none"``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


class Calibrator(ABC):
    @abstractmethod
    def fit(self, scores: np.ndarray, labels: np.ndarray, sample_weight: np.ndarray | None = None) -> Calibrator: ...

    @abstractmethod
    def predict(self, scores: np.ndarray) -> np.ndarray: ...


def _pava(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators: nearest non-decreasing fit to ``y`` (weighted)."""
    blocks: list[list[float]] = []  # [value, weight, count]
    for yi, wi in zip(y, w, strict=True):
        blocks.append([float(yi), float(wi), 1])
        while len(blocks) >= 2 and blocks[-2][0] >= blocks[-1][0]:
            v2, w2, c2 = blocks.pop()
            v1, w1, c1 = blocks.pop()
            nw = w1 + w2
            blocks.append([(v1 * w1 + v2 * w2) / nw, nw, c1 + c2])
    out = np.empty(int(sum(b[2] for b in blocks)))
    i = 0
    for v, _, c in blocks:
        out[i : i + int(c)] = v
        i += int(c)
    return out


class IsotonicCalibrator(Calibrator):
    """Monotonic calibration via pool-adjacent-violators + linear interpolation."""

    def fit(self, scores, labels, sample_weight=None) -> IsotonicCalibrator:
        s = np.asarray(scores, dtype=float)
        y = np.asarray(labels, dtype=float)
        w = np.ones_like(s) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        order = np.argsort(s, kind="mergesort")
        s, y, w = s[order], y[order], w[order]
        fit = _pava(y, w)
        # Collapse duplicate scores so np.interp sees a strictly increasing grid.
        uniq, inv = np.unique(s, return_inverse=True)
        agg = np.zeros(len(uniq))
        cnt = np.zeros(len(uniq))
        np.add.at(agg, inv, fit)
        np.add.at(cnt, inv, 1.0)
        self.x_ = uniq
        self.y_ = agg / cnt
        return self

    def predict(self, scores) -> np.ndarray:
        s = np.asarray(scores, dtype=float)
        if len(self.x_) == 1:
            return np.full_like(s, self.y_[0])
        return np.interp(s, self.x_, self.y_, left=self.y_[0], right=self.y_[-1])


class SigmoidCalibrator(Calibrator):
    """Platt scaling: fit ``P = σ(a·s + b)`` by weighted logistic regression."""

    def __init__(self, lr: float = 0.5, n_iter: int = 500) -> None:
        self.lr = lr
        self.n_iter = n_iter

    def fit(self, scores, labels, sample_weight=None) -> SigmoidCalibrator:
        s = np.asarray(scores, dtype=float)
        y = np.asarray(labels, dtype=float)
        w = np.ones_like(s) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        w = w / w.sum()
        a, b = 1.0, 0.0
        for _ in range(self.n_iter):
            p = sigmoid(a * s + b)
            g = (p - y) * w
            a -= self.lr * float((g * s).sum())
            b -= self.lr * float(g.sum())
        self.a_, self.b_ = a, b
        return self

    def predict(self, scores) -> np.ndarray:
        return sigmoid(self.a_ * np.asarray(scores, dtype=float) + self.b_)


class IdentityCalibrator(Calibrator):
    def fit(self, scores, labels, sample_weight=None) -> IdentityCalibrator:
        return self

    def predict(self, scores) -> np.ndarray:
        return np.asarray(scores, dtype=float)


def make_calibrator(method: str) -> Calibrator:
    """Factory: ``'isotonic'`` | ``'sigmoid'`` (Platt) | ``'none'``."""
    method = (method or "none").lower()
    if method == "isotonic":
        return IsotonicCalibrator()
    if method in ("sigmoid", "platt"):
        return SigmoidCalibrator()
    if method in ("none", "identity"):
        return IdentityCalibrator()
    raise ValueError(f"unknown calibration method '{method}'")
