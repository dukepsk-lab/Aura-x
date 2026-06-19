"""Baseline strategies — the comparators every metric is reported against.

The source PPO "edge" looked far weaker once read against its own Buy-and-Hold
baseline. So a strategy is only interesting if it beats all three "no-skill"
comparators on the *same* event universe, after costs:

* **majority-class** — always bet the most common label direction;
* **Buy-and-Hold**   — always long the underlying;
* **random-entry**   — random ±1 sides (averaged over seeds) ≈ zero edge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import CostModel, strategy_returns
from .metrics import sharpe_ratio


def majority_class_side(labels: pd.Series) -> int:
    """The most common non-flat label direction (+1/−1), or +1 on a tie/empty."""
    nonzero = labels[labels != 0]
    if nonzero.empty:
        return 1
    return 1 if (nonzero > 0).sum() >= (nonzero < 0).sum() else -1


def majority_baseline_returns(
    labels: pd.Series, ret: pd.Series, *, cost_model: CostModel | None = None
) -> pd.Series:
    """Always bet the majority direction."""
    s = majority_class_side(labels)
    side = pd.Series(float(s), index=ret.index)
    return strategy_returns(side, ret, cost_model=cost_model)


def buy_and_hold_returns(close: pd.Series) -> pd.Series:
    """Always-long close-to-close returns of the underlying (the canonical B&H)."""
    return close.pct_change().dropna()


def random_entry_sharpe(
    ret: pd.Series,
    *,
    n_seeds: int = 25,
    base_seed: int = 0,
    cost_model: CostModel | None = None,
    periods_per_year: int | None = None,
) -> float:
    """Mean Sharpe of random ±1 entries over ``n_seeds`` draws (≈ no-skill control)."""
    sharpes = []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        side = pd.Series(rng.choice([-1.0, 1.0], size=len(ret)), index=ret.index)
        r = strategy_returns(side, ret, cost_model=cost_model)
        sharpes.append(sharpe_ratio(r.to_numpy(), periods_per_year))
    return float(np.nanmean(sharpes))


def baseline_sharpes(
    labels: pd.Series,
    ret: pd.Series,
    close: pd.Series | None = None,
    *,
    cost_model: CostModel | None = None,
    periods_per_year: int | None = None,
) -> dict[str, float]:
    """Sharpe of all three baselines on the same event universe."""
    out = {
        "majority": sharpe_ratio(
            majority_baseline_returns(labels, ret, cost_model=cost_model).to_numpy(),
            periods_per_year,
        ),
        "random": random_entry_sharpe(
            ret, cost_model=cost_model, periods_per_year=periods_per_year
        ),
    }
    if close is not None:
        out["buy_and_hold"] = sharpe_ratio(buy_and_hold_returns(close).to_numpy(), periods_per_year)
    return out
