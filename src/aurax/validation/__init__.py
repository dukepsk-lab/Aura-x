"""Validation — the gatekeeper. Nothing reaches Layer 7 without clearing it.

* :class:`CombinatorialPurgedCV`  CPCV with purge + embargo
* :func:`deflated_sharpe_ratio`   multiple-testing-honest Sharpe
* cost-adjusted, baseline-relative helpers

Go/No-Go: live only if cost-adjusted, baseline-relative, deflated performance
holds across folds *and* walk-forward. Otherwise, back to the drawing board —
not to a bigger position.
"""

from __future__ import annotations

from .cpcv import CombinatorialPurgedCV, apply_embargo, purge_train_times
from .metrics import (
    baseline_relative,
    cost_adjusted_returns,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)

__all__ = [
    "CombinatorialPurgedCV",
    "purge_train_times",
    "apply_embargo",
    "sharpe_ratio",
    "cost_adjusted_returns",
    "probabilistic_sharpe_ratio",
    "expected_max_sharpe",
    "deflated_sharpe_ratio",
    "baseline_relative",
]
