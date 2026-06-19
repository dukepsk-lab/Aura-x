"""Triple-Barrier labeling with ATR-scaled barriers (training-time only).

The explicit join between the ATR pillar and the labeling pillar: barriers are
**ATR multiples**, not fixed pips, so labels adapt to volatility.

For each event at ``t0`` (entry = close at ``t0``):

* **upper / TP** barrier at ``entry + tp_mult · ATR(t0)``
* **lower / SL** barrier at ``entry − sl_mult · ATR(t0)``
* **vertical** barrier ``vertical_bars`` ahead (timeout)

We walk forward and record which barrier is touched *first* using intrabar
high/low (more faithful than close-only). If both barriers fall inside one H4
bar, order is unknown at this resolution, so we break the tie pessimistically
(``tie_break='sl'``) — a known hook for later refinement with the M15 execution
feed (Layer 0 already stores it).

Two modes:

* **primary** (``side=None``): label = direction of the first barrier
  (+1 up / −1 down / 0 timeout) — trains the L3 SIDE model.
* **meta** (``side`` provided): label = 1 if the primary's bet wins
  (TP hit, or timeout with ``ret·side > min_ret``) else 0 — trains the L5 TRUST
  model on out-of-fold primary predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..enums import BarrierTouch, Label

LABEL_COLUMNS = ["t1", "label", "ret", "barrier", "tp_price", "sl_price", "side"]


def triple_barrier_labels(
    bars: pd.DataFrame,
    atr: pd.Series,
    *,
    tp_mult: float = 2.0,
    sl_mult: float = 2.0,
    vertical_bars: int = 12,
    events: pd.DatetimeIndex | None = None,
    side: pd.Series | None = None,
    min_ret: float = 0.0,
    tie_break: str = "sl",
) -> pd.DataFrame:
    """Label ``events`` by the triple-barrier method. Returns a frame indexed by
    event time with columns ``LABEL_COLUMNS``.

    ``atr`` must be aligned to ``bars`` (e.g. ``aurax.l1_features.atr(bars, n)``).
    Events whose ATR is NaN (warm-up) are skipped.
    """
    if not {"high", "low", "close"}.issubset(bars.columns):
        raise ValueError("bars must contain high/low/close")
    if tie_break not in ("sl", "tp"):
        raise ValueError("tie_break must be 'sl' or 'tp'")

    index = bars.index
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    atr_arr = atr.reindex(index).to_numpy(dtype=float)
    pos = {ts: i for i, ts in enumerate(index)}

    event_index = index if events is None else pd.DatetimeIndex(events)
    side_arr = None if side is None else side.reindex(event_index).to_numpy(dtype=float)
    meta_mode = side is not None

    records: list[dict] = []
    n = len(index)
    for k, ts in enumerate(event_index):
        i0 = pos.get(ts)
        if i0 is None or i0 + 1 >= n:
            continue
        a = atr_arr[i0]
        if not np.isfinite(a) or a <= 0:
            continue

        entry = close[i0]
        s = 1.0 if not meta_mode else float(side_arr[k])
        if meta_mode and s == 0:
            continue

        # TP/SL price levels (oriented by side; multipliers attach to TP vs SL).
        if s >= 0:
            tp_price, sl_price = entry + tp_mult * a, entry - sl_mult * a
        else:
            tp_price, sl_price = entry - tp_mult * a, entry + sl_mult * a

        i_end = min(i0 + vertical_bars, n - 1)
        outcome = BarrierTouch.VERTICAL
        t1_idx = i_end
        for j in range(i0 + 1, i_end + 1):
            if s >= 0:
                hit_tp, hit_sl = high[j] >= tp_price, low[j] <= sl_price
            else:
                hit_tp, hit_sl = low[j] <= tp_price, high[j] >= sl_price
            if hit_tp and hit_sl:
                outcome = BarrierTouch.SL if tie_break == "sl" else BarrierTouch.TP
                t1_idx = j
                break
            if hit_tp:
                outcome, t1_idx = BarrierTouch.TP, j
                break
            if hit_sl:
                outcome, t1_idx = BarrierTouch.SL, j
                break

        # Exit at the barrier level if touched, else at the timeout close.
        if outcome is BarrierTouch.TP:
            exit_price = tp_price
        elif outcome is BarrierTouch.SL:
            exit_price = sl_price
        else:
            exit_price = close[t1_idx]
        raw_ret = exit_price / entry - 1.0

        if meta_mode:
            signed = raw_ret * s
            won = outcome is BarrierTouch.TP or (
                outcome is BarrierTouch.VERTICAL and signed > min_ret
            )
            label = int(won)  # 1 take / 0 pass
            ret_out = signed
        else:
            if outcome is BarrierTouch.TP:
                label = int(Label.UP)
            elif outcome is BarrierTouch.SL:
                label = int(Label.DOWN)
            else:
                label = int(Label.NEUTRAL)
            ret_out = raw_ret

        records.append(
            {
                "ts": ts,
                "t1": index[t1_idx],
                "label": label,
                "ret": ret_out,
                "barrier": outcome.value,
                "tp_price": tp_price,
                "sl_price": sl_price,
                "side": (s if meta_mode else np.nan),
            }
        )

    if not records:
        return pd.DataFrame(columns=LABEL_COLUMNS, index=pd.DatetimeIndex([], name="ts"))
    out = pd.DataFrame.from_records(records, index="ts")
    out.index.name = "ts"
    return out[LABEL_COLUMNS]
