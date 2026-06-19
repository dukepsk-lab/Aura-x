"""Event-level, cost-adjusted backtest over Layer-4 labeled events.

Triple-barrier labels are discrete bets: each non-flat event is a round-trip
trade to a barrier, and its realised return is the label's ``ret`` column. A
strategy's per-event return is therefore ``predicted_side · ret`` minus realistic
costs (spread + slippage + commission). "An edge that only exists gross is not an
edge", so costs are first-class here.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class CostModel:
    """Round-trip transaction costs, expressed so they convert to return units."""

    spread_pips: float = 1.0
    slippage_pips: float = 0.5
    pip_size: float = 0.0001
    commission_ret: float = 0.0  # fixed commission already in return units

    def per_trade_cost(self, entry_price: pd.Series | float) -> pd.Series | float:
        """Round-trip cost as a fraction of notional.

        ``(spread + slippage) · pip_size / price`` converts a pip cost into a
        return; FX majors sit near 1.0 so a missing price defaults there.
        """
        pip_cost = (self.spread_pips + self.slippage_pips) * self.pip_size
        return pip_cost / entry_price + self.commission_ret


def strategy_returns(
    side: pd.Series,
    ret: pd.Series,
    *,
    entry_price: pd.Series | None = None,
    cost_model: CostModel | None = None,
) -> pd.Series:
    """Per-event cost-adjusted strategy returns = ``side · ret − cost·traded``.

    ``side`` ∈ {−1, 0, +1} (0 = stand aside, no cost). ``ret`` is the L4 realised
    return to barrier. Costs apply only to non-flat (actually traded) events.
    """
    side = side.reindex(ret.index).fillna(0.0)
    gross = side * ret
    if cost_model is None:
        return gross
    price = entry_price.reindex(ret.index) if entry_price is not None else 1.0
    cost = cost_model.per_trade_cost(price) * (side != 0).astype(float)
    return gross - cost


def equity_curve(returns: pd.Series) -> pd.Series:
    """Additive equity curve (sum of per-event returns)."""
    return returns.cumsum()


def turnover(side: pd.Series) -> float:
    """Fraction of events that are actually traded (non-flat)."""
    if len(side) == 0:
        return 0.0
    return float((side != 0).mean())
