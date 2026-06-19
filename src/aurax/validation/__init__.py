"""Validation — the gatekeeper. Nothing reaches Layer 7 without clearing it.

Splitters (leakage control):
* :class:`CombinatorialPurgedCV`  CPCV with purge + embargo
* :class:`PurgedKFold`            purged k-fold → out-of-fold predictions (L5 bridge)
* :class:`WalkForwardSplit` / :func:`holdout_split`  forward + never-touched tail

Backtest & baselines (cost-adjusted, baseline-relative):
* :class:`CostModel`, :func:`strategy_returns`   event-level PnL after costs
* :func:`baseline_sharpes`                        majority / Buy-and-Hold / random

Honesty & orchestration:
* :func:`deflated_sharpe_ratio`   multiple-testing-honest Sharpe
* :func:`run_validation` → :class:`ValidationReport`   the full §5 Go/No-Go
* :func:`oof_predict`             out-of-fold predictions for the meta-model

Go/No-Go: live only if cost-adjusted, baseline-relative, deflated performance
holds across CPCV folds *and* walk-forward *and* the holdout. Otherwise, back to
the drawing board — not to a bigger position.
"""

from __future__ import annotations

from .backtest import CostModel, equity_curve, strategy_returns, turnover
from .baselines import (
    baseline_sharpes,
    buy_and_hold_returns,
    majority_baseline_returns,
    majority_class_side,
    random_entry_sharpe,
)
from .cpcv import CombinatorialPurgedCV, PurgedKFold, apply_embargo, purge_train_times
from .harness import (
    Estimator,
    ValidationConfig,
    ValidationReport,
    oof_predict,
    run_validation,
)
from .metrics import (
    baseline_relative,
    cost_adjusted_returns,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    max_drawdown,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
    sortino_ratio,
)
from .walkforward import WalkForwardSplit, holdout_split

__all__ = [
    # splitters
    "CombinatorialPurgedCV",
    "PurgedKFold",
    "WalkForwardSplit",
    "holdout_split",
    "purge_train_times",
    "apply_embargo",
    # backtest & baselines
    "CostModel",
    "strategy_returns",
    "equity_curve",
    "turnover",
    "baseline_sharpes",
    "majority_class_side",
    "majority_baseline_returns",
    "buy_and_hold_returns",
    "random_entry_sharpe",
    # metrics
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "cost_adjusted_returns",
    "probabilistic_sharpe_ratio",
    "expected_max_sharpe",
    "deflated_sharpe_ratio",
    "baseline_relative",
    # harness
    "run_validation",
    "oof_predict",
    "ValidationReport",
    "ValidationConfig",
    "Estimator",
]
