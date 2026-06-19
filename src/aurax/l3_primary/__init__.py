"""Layer 3 — Primary Signal Model (SIDE): LightGBM/CatBoost ensemble, high recall.

* :class:`PrimarySignalModel` / :class:`PrimaryConfig`  regime-conditional ensemble
* :class:`SideModel`  member interface (implement + register to add a CNN/PatchTST)
* :class:`LogisticSide` / :class:`LightGBMSide` / :class:`CatBoostSide`  members
* :func:`probas_to_side`  recall-oriented decision rule
"""

from __future__ import annotations

from .base import CLASSES, PROBA_COLUMNS, SideModel, probas_to_side
from .ensemble import PrimaryConfig, PrimarySignalModel
from .members import (
    MEMBER_REGISTRY,
    CatBoostSide,
    LightGBMSide,
    LogisticSide,
    available_backends,
    build_member,
)

__all__ = [
    "PrimarySignalModel",
    "PrimaryConfig",
    "SideModel",
    "probas_to_side",
    "CLASSES",
    "PROBA_COLUMNS",
    "LogisticSide",
    "LightGBMSide",
    "CatBoostSide",
    "MEMBER_REGISTRY",
    "build_member",
    "available_backends",
]
