"""Layer 6 — Risk & Position Sizing (+ L6b Allocation).

* :class:`RiskManager` / :class:`RiskConfig`  the portfolio risk gateway
* :class:`SizingRequest` / :class:`SizingDecision` / :class:`SizingReason`
* pure formulas: :func:`position_size_atr`, :func:`confidence_multiplier`,
  :func:`aggregate_correlated_risk`, :func:`apply_correlation_cap`
"""

from __future__ import annotations

from .allocation import AllocationConfig, DirichletAllocator, reward
from .sizing import (
    RiskConfig,
    RiskManager,
    SizingDecision,
    SizingReason,
    SizingRequest,
    aggregate_correlated_risk,
    apply_correlation_cap,
    confidence_multiplier,
    correlation_matrix,
    position_size_atr,
    round_to_lot,
)

__all__ = [
    "RiskManager",
    "RiskConfig",
    "SizingRequest",
    "SizingDecision",
    "SizingReason",
    "position_size_atr",
    "confidence_multiplier",
    "apply_correlation_cap",
    "aggregate_correlated_risk",
    "correlation_matrix",
    "round_to_lot",
    "DirichletAllocator",
    "AllocationConfig",
    "reward",
]
