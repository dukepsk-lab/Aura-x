"""Layer 3 — Primary Signal Model (SIDE).

Predicts **direction only** (long / short / flat) and is tuned for **high
recall** — surface every plausible opportunity; precision is delegated to the
Layer 5 meta-model. Backbone: a **LightGBM / CatBoost ensemble** (with a
dependency-free logistic fallback), blended **regime-conditionally** — Layer 2
selects the per-member weights, so momentum-leaning members can dominate in trend
regimes and mean-reversion-leaning members in ranging regimes.

Trained on Layer 4 direction labels. The Layer 5 meta-model is later trained on
this model's **out-of-fold** predictions (``aurax.validation.oof_predict``),
never in-sample.

The model is sklearn-compatible (``fit`` / ``predict`` → side ints), so it drops
straight into the validation gate and ``oof_predict``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .base import PROBA_COLUMNS, SideModel, normalize_rows, probas_to_side
from .members import available_backends, build_member


@dataclass
class PrimaryConfig:
    """Ensemble members, per-regime weight overrides, and the recall threshold."""

    members: dict[str, float] = field(default_factory=lambda: {"lightgbm": 0.6, "catboost": 0.4})
    member_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    regime_overrides: dict[str, dict[str, float]] = field(default_factory=dict)
    flat_threshold: float = 0.15

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> PrimaryConfig:
        p = params.get("primary", {})
        return cls(
            members=p.get("members", {"lightgbm": 0.6, "catboost": 0.4}),
            member_params=p.get("member_params", {}),
            regime_overrides=p.get("regime_overrides", {}),
            flat_threshold=p.get("flat_threshold", 0.15),
        )

    @classmethod
    def lightweight(cls) -> PrimaryConfig:
        """Single dependency-free logistic member — runs without ``[models]``."""
        return cls(members={"logistic": 1.0})

    @classmethod
    def from_available(cls, params: dict[str, Any] | None = None) -> PrimaryConfig:
        """Config using the configured members if their backends are installed,
        else falling back to the logistic member."""
        cfg = cls.from_params(params or {})
        backends = set(available_backends())
        if not set(cfg.members).issubset(backends):
            return cls.lightweight()
        return cfg


def _normalize(weights: dict[str, float], names: list[str]) -> np.ndarray:
    vec = np.array([max(0.0, weights.get(n, 0.0)) for n in names], dtype=float)
    total = vec.sum()
    return vec / total if total > 0 else np.full(len(names), 1.0 / len(names))


class PrimarySignalModel:
    """Regime-conditional, high-recall direction ensemble (SIDE)."""

    def __init__(self, config: PrimaryConfig | None = None) -> None:
        self.config = config or PrimaryConfig()
        self._names = list(self.config.members)
        self.members_: dict[str, SideModel] = {}

    # --- training ------------------------------------------------------------
    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None = None
    ) -> PrimarySignalModel:
        """Fit every member globally on the L4 direction labels."""
        self.members_ = {}
        for name in self._names:
            member = build_member(name, **self.config.member_params.get(name, {}))
            member.fit(X, y, sample_weight=sample_weight)
            self.members_[name] = member
        return self

    # --- inference -----------------------------------------------------------
    def _regime_weights(self, regime: object) -> np.ndarray:
        override = self.config.regime_overrides.get(str(regime))
        base = override if override else self.config.members
        return _normalize(base, self._names)

    def predict_proba(
        self, X: pd.DataFrame, regimes: pd.Series | None = None
    ) -> pd.DataFrame:
        """Regime-weighted class probabilities ``[p_down, p_flat, p_up]``."""
        if not self.members_:
            raise RuntimeError("PrimarySignalModel is not fitted")
        stack = np.stack(
            [self.members_[n].predict_proba(X).to_numpy() for n in self._names], axis=0
        )  # (M, N, 3)

        if regimes is None:
            w = _normalize(self.config.members, self._names)
            blended = np.tensordot(w, stack, axes=(0, 0))  # (N, 3)
        else:
            reg = regimes.reindex(X.index).to_numpy()
            cache: dict[object, np.ndarray] = {}
            w_rows = np.array(
                [cache.setdefault(r, self._regime_weights(r)) for r in reg]
            )  # (N, M)
            blended = np.einsum("nm,mnk->nk", w_rows, stack)

        frame = pd.DataFrame(blended, index=X.index, columns=list(PROBA_COLUMNS))
        return normalize_rows(frame)

    def predict_side(
        self, X: pd.DataFrame, regimes: pd.Series | None = None
    ) -> pd.Series:
        """Direction decision in ``{-1, 0, +1}`` (high recall via flat_threshold)."""
        return probas_to_side(self.predict_proba(X, regimes), self.config.flat_threshold)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """sklearn-compatible side prediction (global blend) for the validation gate."""
        return self.predict_side(X).to_numpy()
