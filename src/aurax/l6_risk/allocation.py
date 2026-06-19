"""Layer 6b — Allocation (optional, advanced; scaffold).

For 2 instruments, full RL portfolio optimization is overkill — the §6 rules
capture most of the value. But the reward-engineering philosophy is the correct
objective for any allocator::

    maximize  log_return − λ₁·turnover − λ₂·risk(covariance)

When the universe grows (Roadmap v3), promote this to a **Dirichlet-policy PPO**
agent allocating across pairs/sub-strategies under a native simplex constraint
(weights ≥ 0, sum to 1). Until then this stays the explicit objective behind the
§6 sizing rules.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AllocationConfig:
    turnover_penalty: float = 0.001   # λ₁
    risk_penalty: float = 0.5         # λ₂


def reward(
    log_return: float,
    turnover: float,
    covariance_risk: float,
    config: AllocationConfig | None = None,
) -> float:
    """The cost/risk-penalized allocation objective (fully specified, pure)."""
    cfg = config or AllocationConfig()
    return log_return - cfg.turnover_penalty * turnover - cfg.risk_penalty * covariance_risk


class DirichletAllocator:
    """Dirichlet-policy PPO allocator over the instrument simplex (Roadmap v3)."""

    def __init__(self, config: AllocationConfig | None = None) -> None:
        self.config = config or AllocationConfig()

    def allocate(self, expected: np.ndarray, covariance: np.ndarray) -> np.ndarray:
        """Return simplex weights (≥0, sum to 1). TODO(v3)."""
        raise NotImplementedError("Dirichlet-policy allocator lands in Roadmap v3 (L6b).")
