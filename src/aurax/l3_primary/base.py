"""Primary-model primitives — the ``SideModel`` member interface and encoding.

Layer 3 predicts **direction only** as a 3-class problem over
``CLASSES = (-1, 0, +1)`` (down / flat / up). Each ensemble *member* implements
:class:`SideModel` (``fit`` / ``predict_proba``); the ensemble blends them and
applies a recall-oriented decision rule.

Two robustness rules live here so every member obeys them:

* probabilities are always returned over the full 3-class frame
  (:data:`PROBA_COLUMNS`), even if a class was absent from the training fold —
  absent classes get probability 0 (then rows renormalise);
* the up/down **margin** drives the side decision, so a low ``flat_threshold``
  yields high recall (rarely abstain) — precision is delegated to L5.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

#: Canonical class order: down, flat, up.
CLASSES: tuple[int, int, int] = (-1, 0, 1)
PROBA_COLUMNS: tuple[str, str, str] = ("p_down", "p_flat", "p_up")
_CLASS_TO_COL = dict(zip(CLASSES, PROBA_COLUMNS, strict=True))


def encode_labels(y: pd.Series) -> np.ndarray:
    """Validate that labels are in ``{-1, 0, 1}`` and return them as ``int``."""
    arr = np.asarray(y, dtype=int)
    bad = set(np.unique(arr)) - set(CLASSES)
    if bad:
        raise ValueError(f"labels must be in {CLASSES}; got unexpected {sorted(bad)}")
    return arr


def align_proba(
    proba: np.ndarray, classes_present: list[int] | np.ndarray, index: pd.Index
) -> pd.DataFrame:
    """Expand a member's per-class probabilities to the full 3-class frame.

    ``proba`` has one column per class in ``classes_present`` (in that order);
    missing canonical classes are filled with 0 so every member returns
    :data:`PROBA_COLUMNS` in canonical order.
    """
    out = pd.DataFrame(0.0, index=index, columns=list(PROBA_COLUMNS))
    for j, cls in enumerate(classes_present):
        out[_CLASS_TO_COL[int(cls)]] = proba[:, j]
    return out


def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise numerically-stable softmax."""
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def normalize_rows(proba: pd.DataFrame) -> pd.DataFrame:
    """Renormalise each row to sum to 1 (after blending / class-filling)."""
    s = proba.sum(axis=1)
    s = s.where(s != 0, 1.0)
    return proba.div(s, axis=0)


def probas_to_side(proba: pd.DataFrame, flat_threshold: float = 0.15) -> pd.Series:
    """Map class probabilities to a side via the up−down margin (high recall).

    Abstain (0) only when ``|p_up − p_down| < flat_threshold`` — i.e. when there
    is no real directional edge. Lower threshold → more trades surfaced.
    """
    margin = proba["p_up"] - proba["p_down"]
    side = np.sign(margin)
    side[margin.abs() < flat_threshold] = 0
    return pd.Series(side.astype(int), index=proba.index, name="side")


class SideModel(ABC):
    """An ensemble member: fit on (X, y∈{-1,0,1}), output 3-class probabilities."""

    name: str = "member"

    @abstractmethod
    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None = None
    ) -> SideModel: ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a frame with :data:`PROBA_COLUMNS`, indexed like ``X``."""
