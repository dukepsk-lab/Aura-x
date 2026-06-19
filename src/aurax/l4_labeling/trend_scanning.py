"""Trend-scanning labels (López de Prado) — optional complementary target.

For each event ``t0`` we scan forward horizons ``L ∈ [min, max]``, fit an OLS of
log price on time over ``[t0, t0+L]``, and keep the horizon whose slope has the
largest |t-value|. The label is the **sign of that slope** (0 if |t| below
threshold). Unlike triple-barrier (path-aware, asymmetric), this gives a
smooth-trend target well suited to the trend-regime sub-model.

Like all of Layer 4, this is **training-time only** and deliberately looks
forward — that is what a label is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TREND_COLUMNS = ["t1", "label", "ret", "t_value", "horizon"]


def _slope_tvalue(y: np.ndarray) -> tuple[float, float]:
    """OLS slope and its t-statistic for ``y`` regressed on equally spaced time."""
    n = y.size
    if n < 3:
        return 0.0, 0.0
    x = np.arange(n, dtype=float)
    x_dm = x - x.mean()
    sxx = (x_dm**2).sum()
    if sxx == 0:
        return 0.0, 0.0
    slope = (x_dm * (y - y.mean())).sum() / sxx
    intercept = y.mean() - slope * x.mean()
    resid = y - (intercept + slope * x)
    dof = n - 2
    s2 = (resid**2).sum() / dof
    se = np.sqrt(s2 / sxx)
    if se == 0:
        return slope, 0.0
    return slope, slope / se


def trend_scanning_labels(
    close: pd.Series,
    *,
    min_horizon: int = 5,
    max_horizon: int = 20,
    t_value_threshold: float = 2.0,
    events: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Label events by the maximum-|t| forward trend. Returns ``TREND_COLUMNS``."""
    log_price = np.log(close).to_numpy(dtype=float)
    index = close.index
    pos = {ts: i for i, ts in enumerate(index)}
    event_index = index if events is None else pd.DatetimeIndex(events)
    n = len(index)

    records: list[dict] = []
    for ts in event_index:
        i0 = pos.get(ts)
        if i0 is None or i0 + min_horizon >= n:
            continue
        best_t, best_slope, best_L = 0.0, 0.0, min_horizon
        for L in range(min_horizon, max_horizon + 1):
            i_end = i0 + L
            if i_end >= n:
                break
            slope, tval = _slope_tvalue(log_price[i0 : i_end + 1])
            if abs(tval) > abs(best_t):
                best_t, best_slope, best_L = tval, slope, L
        i_end = min(i0 + best_L, n - 1)
        label = int(np.sign(best_slope)) if abs(best_t) >= t_value_threshold else 0
        records.append(
            {
                "ts": ts,
                "t1": index[i_end],
                "label": label,
                "ret": float(log_price[i_end] - log_price[i0]),
                "t_value": float(best_t),
                "horizon": int(best_L),
            }
        )

    if not records:
        return pd.DataFrame(columns=TREND_COLUMNS, index=pd.DatetimeIndex([], name="ts"))
    out = pd.DataFrame.from_records(records, index="ts")
    out.index.name = "ts"
    return out[TREND_COLUMNS]
