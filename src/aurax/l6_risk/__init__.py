"""Layer 6 — Risk & Position Sizing (+ L6b Allocation, Roadmap v3).

* :class:`RiskManager` / :class:`RiskConfig`  the portfolio risk gateway
* :class:`SizingRequest` / :class:`SizingDecision` / :class:`SizingReason`
* pure formulas: :func:`position_size_atr`, :func:`confidence_multiplier`,
  :func:`aggregate_correlated_risk`, :func:`apply_correlation_cap`
* :class:`DirichletAllocator` / :class:`AllocationConfig`  L6b's PPO allocator
  across the instrument simplex, and its pure reward components
  (:func:`reward`, :func:`covariance_risk`, :func:`portfolio_turnover`,
  :func:`portfolio_log_return`)
"""

from __future__ import annotations

from .allocation import (
    AllocationConfig,
    DirichletAllocator,
    covariance_risk,
    portfolio_log_return,
    portfolio_turnover,
    reward,
)
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
    "covariance_risk",
    "portfolio_turnover",
    "portfolio_log_return",
]
