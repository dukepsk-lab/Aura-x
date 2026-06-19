"""Combinatorial Purged Cross-Validation (López de Prado, AFML ch. 7).

CPCV with **purging + embargo** kills leakage from overlapping H4 labels — the
failure mode that quietly inflates backtests. Implemented here as pure index
logic (no model dependency) so it can gate any estimator.

* **Purge:** drop train observations whose label span ``[t0, t1]`` overlaps the
  test span (AFML 7.1).
* **Embargo:** additionally drop train observations falling within a short
  window *after* each test block (AFML 7.2), removing serial-correlation leakage.
* **Combinatorial:** test on every size-``k`` combination of ``N`` groups,
  yielding ``C(N, k)`` train/test splits and many backtest paths.
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import combinations

import numpy as np
import pandas as pd


def purge_train_times(t1: pd.Series, test_t1: pd.Series) -> pd.Series:
    """Drop train labels whose span overlaps any test label span (AFML 7.1).

    ``t1``/``test_t1`` are Series indexed by event start (``t0``) with value the
    event end. Returns the surviving train spans.
    """
    train = t1.copy()
    for t0, t1_i in test_t1.items():
        starts_within = train[(t0 <= train.index) & (train.index <= t1_i)].index
        ends_within = train[(t0 <= train) & (train <= t1_i)].index
        envelops = train[(train.index <= t0) & (t1_i <= train)].index
        train = train.drop(starts_within.union(ends_within).union(envelops))
    return train


def apply_embargo(
    train: pd.Series, test_t1: pd.Series, bar_index: pd.DatetimeIndex, embargo_bars: int
) -> pd.Series:
    """Drop train observations within ``embargo_bars`` after each test block."""
    if embargo_bars <= 0 or train.empty:
        return train
    drop: set = set()
    pos = {ts: i for i, ts in enumerate(bar_index)}
    # Iterate the Series directly: this preserves tz-aware Timestamps, whereas
    # `.values` would coerce to tz-naive numpy.datetime64 and miss the dict.
    for t1_i in test_t1:
        end_pos = pos.get(t1_i)
        if end_pos is None:
            continue
        emb_end = min(end_pos + embargo_bars, len(bar_index) - 1)
        embargo_span = bar_index[end_pos : emb_end + 1]
        drop.update(ts for ts in train.index if ts in embargo_span)
    return train.drop(pd.Index(sorted(drop)))


class CombinatorialPurgedCV:
    """Combinatorial Purged CV splitter.

    Parameters mirror ``config/default.yaml`` ``validation``: ``n_groups``,
    ``n_test_groups`` and ``embargo_pct`` (fraction of the sample count).
    """

    def __init__(
        self, n_groups: int = 6, n_test_groups: int = 2, embargo_pct: float = 0.01
    ) -> None:
        if not 1 <= n_test_groups < n_groups:
            raise ValueError("require 1 <= n_test_groups < n_groups")
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.embargo_pct = embargo_pct

    @property
    def n_splits(self) -> int:
        from math import comb

        return comb(self.n_groups, self.n_test_groups)

    def split(self, t1: pd.Series) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_pos, test_pos)`` integer-position arrays per combination.

        ``t1`` is a Series indexed by event time (``t0``, ascending) with value
        the label end time — exactly the ``t1`` column from Layer 4.
        """
        n = len(t1)
        bar_index = t1.index
        embargo_bars = int(n * self.embargo_pct)
        group_bounds = np.array_split(np.arange(n), self.n_groups)
        pos_of = {ts: i for i, ts in enumerate(bar_index)}

        for combo in combinations(range(self.n_groups), self.n_test_groups):
            test_pos = np.concatenate([group_bounds[g] for g in combo])
            test_pos.sort()
            test_t1 = t1.iloc[test_pos]

            surviving = purge_train_times(t1, test_t1)
            surviving = apply_embargo(surviving, test_t1, bar_index, embargo_bars)
            train_pos = np.array(
                sorted(pos_of[ts] for ts in surviving.index), dtype=int
            )
            yield train_pos, test_pos
