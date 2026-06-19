"""Layer 6 — Risk & Position Sizing (+ L6b Allocation)."""

from __future__ import annotations

from .allocation import AllocationConfig, DirichletAllocator, reward
from .sizing import (
    RiskConfig,
    RiskManager,
    apply_correlation_cap,
    confidence_multiplier,
    position_size_atr,
)

__all__ = [
    "position_size_atr",
    "confidence_multiplier",
    "apply_correlation_cap",
    "RiskManager",
    "RiskConfig",
    "DirichletAllocator",
    "AllocationConfig",
    "reward",
]
