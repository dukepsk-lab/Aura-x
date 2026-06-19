"""Sample-uniqueness weighting (López de Prado, AFML ch. 4).

Overlapping H4 labels share information: a label spanning ``[t0, t1]`` overlaps
every other label whose span intersects it. Treating them as i.i.d. silently
inflates backtests. We down-weight overlapping samples by their **average
uniqueness**, optionally weight by **return attribution**, and optionally apply a
**linear time-decay** so stale observations matter less.

Weights are normalised to mean 1.0 so they plug straight into
``sample_weight=`` of LightGBM/CatBoost/sklearn without rescaling the loss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def num_concurrent_events(bar_index: pd.DatetimeIndex, t1: pd.Series) -> pd.Series:
    """Count, per bar, how many label spans ``[t0, t1]`` are live at that bar."""
    t1 = t1.dropna()
    if t1.empty:
        return pd.Series(dtype=float)
    span = bar_index[(bar_index >= t1.index.min()) & (bar_index <= t1.max())]
    count = pd.Series(0.0, index=span)
    for t0, t1_i in t1.items():
        count.loc[t0:t1_i] += 1.0
    return count


def average_uniqueness(t1: pd.Series, concurrency: pd.Series) -> pd.Series:
    """Average uniqueness of each label = mean of ``1/concurrency`` over its span."""
    out = pd.Series(index=t1.index, dtype=float)
    for t0, t1_i in t1.items():
        c = concurrency.loc[t0:t1_i]
        out.loc[t0] = (1.0 / c).mean() if len(c) else np.nan
    return out


def return_attribution(t1: pd.Series, concurrency: pd.Series, close: pd.Series) -> pd.Series:
    """Weight by |Σ (log-return / concurrency)| over each label's span (AFML 4.10)."""
    log_ret = np.log(close).diff()
    out = pd.Series(index=t1.index, dtype=float)
    for t0, t1_i in t1.items():
        r = log_ret.loc[t0:t1_i]
        c = concurrency.loc[t0:t1_i]
        out.loc[t0] = (r / c).sum()
    return out.abs()


def time_decay(av_uniqueness: pd.Series, last_weight: float = 0.5) -> pd.Series:
    """Linear time-decay (AFML 4.10). Newest sample weight 1.0; oldest →
    ``last_weight`` (in [0,1]; <0 truncates the oldest to zero weight)."""
    decay = av_uniqueness.sort_index().cumsum()
    total = decay.iloc[-1]
    slope = (
        (1.0 - last_weight) / total
        if last_weight >= 0
        else 1.0 / ((last_weight + 1.0) * total)
    )
    const = 1.0 - slope * total
    weights = const + slope * decay
    weights[weights < 0] = 0.0
    return weights


def sample_weights(
    t1: pd.Series,
    bar_index: pd.DatetimeIndex,
    *,
    close: pd.Series | None = None,
    by_return: bool = False,
    apply_time_decay: bool = True,
    time_decay_last_weight: float = 0.5,
    normalize: bool = True,
) -> pd.Series:
    """Compute per-label training weights from label spans.

    ``t1`` is a Series indexed by event start (``t0``) whose values are the
    barrier-touch / timeout time. Returns weights aligned to ``t1.index``.
    """
    if t1.empty:
        return pd.Series(dtype=float)

    concurrency = num_concurrent_events(bar_index, t1)
    if by_return:
        if close is None:
            raise ValueError("by_return=True requires the close series")
        base = return_attribution(t1, concurrency, close)
    else:
        base = average_uniqueness(t1, concurrency)

    weights = base.copy()
    if apply_time_decay:
        weights = weights * time_decay(average_uniqueness(t1, concurrency), time_decay_last_weight)

    weights = weights.fillna(0.0)
    if normalize and weights.sum() > 0:
        weights = weights * (len(weights) / weights.sum())  # mean → 1.0
    return weights
