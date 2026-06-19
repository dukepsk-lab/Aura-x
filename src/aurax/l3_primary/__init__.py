"""Layer 3 — Primary Signal Model (SIDE): CNN + LightGBM/CatBoost ensemble."""

from __future__ import annotations

from .ensemble import PrimaryConfig, PrimarySignalModel

__all__ = ["PrimarySignalModel", "PrimaryConfig"]
