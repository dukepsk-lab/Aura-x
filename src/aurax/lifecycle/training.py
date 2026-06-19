"""Training pipeline + model persistence (Roadmap v2 — lifecycle).

Fits the full decision stack (L3 primary + L5 meta) into one versioned bundle and
persists it. This is the reproducible training entry point the retrain
orchestrator drives.

Persistence uses pickle, which covers the logistic / LightGBM / CatBoost members.
Deep (torch) members hold locally-defined modules that don't pickle; persisting
those would use ``torch.save`` on the state dict (a documented follow-up).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..l3_primary import PrimaryConfig, PrimarySignalModel
from ..l5_meta import MetaConfig, MetaLabelModel, train_meta_labeler


@dataclass
class TrainedModel:
    """A versioned, persistable bundle of the fitted decision stack."""

    primary: PrimarySignalModel
    meta: MetaLabelModel | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict_side(self, X: pd.DataFrame, regimes: pd.Series | None = None) -> pd.Series:
        """Primary direction (the meta gate is applied by the live wiring / L6)."""
        return self.primary.predict_side(X, regimes)


class TrainingPipeline:
    """Fit primary + meta into a :class:`TrainedModel` and save/load it."""

    def __init__(
        self,
        primary_config: PrimaryConfig | None = None,
        meta_config: MetaConfig | None = None,
        *,
        feature_set: str = "v1",
    ) -> None:
        self.primary_config = primary_config or PrimaryConfig.lightweight()
        self.meta_config = meta_config or MetaConfig()
        self.feature_set = feature_set

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        t1: pd.Series,
        ret: pd.Series | None = None,
        regimes: pd.Series | None = None,
        context: pd.DataFrame | None = None,
        sample_weight: pd.Series | None = None,
        fit_meta: bool = True,
    ) -> TrainedModel:
        """Fit the primary on all data and (optionally) the meta on OOF predictions."""
        primary = PrimarySignalModel(self.primary_config).fit(X, y.astype(int), sample_weight=sample_weight)

        meta = None
        tau = None
        if fit_meta:
            res = train_meta_labeler(
                lambda: PrimarySignalModel(self.primary_config),
                X, y, t1=t1, regimes=regimes, context=context,
                sample_weight=sample_weight, ret=ret, meta_config=self.meta_config,
            )
            meta = res["meta_model"]
            tau = res["tau"]

        metadata = {
            "version": datetime.now(UTC).strftime("%Y%m%d%H%M%S"),
            "trained_at": datetime.now(UTC).isoformat(),
            "n_samples": int(len(X)),
            "feature_set": self.feature_set,
            "primary_members": list(self.primary_config.members),
            "tau": tau,
        }
        return TrainedModel(primary=primary, meta=meta, metadata=metadata)

    @staticmethod
    def save(model: TrainedModel, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(model, fh)
        return path

    @staticmethod
    def load(path: str | Path) -> TrainedModel:
        with Path(path).open("rb") as fh:
            return pickle.load(fh)
