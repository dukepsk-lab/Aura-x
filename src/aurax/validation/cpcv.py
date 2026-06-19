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


def _merge_intervals(
    starts: np.ndarray, ends: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Merge possibly-overlapping ``[start, end]`` intervals into disjoint ones."""
    order = np.argsort(starts, kind="mergesort")
    s, e = starts[order], ends[order]
    cummax_e = np.maximum.accumulate(e)
    new_group = np.empty(len(s), dtype=bool)
    new_group[0] = True
    new_group[1:] = s[1:] > cummax_e[:-1]  # gap → start a new merged interval
    seg_starts = np.flatnonzero(new_group)
    merged_starts = s[seg_starts]
    # Per-group max end via reduceat (segments are contiguous after the sort).
    # NB: do NOT use np.empty + maximum.at here — uninitialised datetime64 memory
    # would survive the max and silently mark every interval as overlapping.
    merged_ends = np.maximum.reduceat(e, seg_starts)
    return merged_starts, merged_ends


def purge_train_times(t1: pd.Series, test_t1: pd.Series) -> pd.Series:
    """Drop train labels whose span overlaps any test label span (AFML 7.1).

    ``t1``/``test_t1`` are Series indexed by event start (``t0``) with value the
    event end. Vectorised via interval-merge + stabbing — O(n log n), and (unlike
    a position bitmap) it needs no shared index, so walk-forward splits work too.
    """
    if t1.empty or test_t1.empty:
        return t1
    a0 = t1.index.values  # train starts (datetime64)
    a1 = t1.values  # train ends
    ms, me = _merge_intervals(test_t1.index.values, test_t1.values)
    # The only merged interval that can overlap [a0, a1] is the rightmost whose
    # start ≤ a1; it overlaps iff its end ≥ a0 (disjoint, sorted invariant).
    idx = np.searchsorted(ms, a1, side="right") - 1
    overlap = np.zeros(len(a0), dtype=bool)
    valid = idx >= 0
    overlap[valid] = me[idx[valid]] >= a0[valid]
    return t1[~overlap]


def apply_embargo(
    train: pd.Series, test_t1: pd.Series, bar_index: pd.DatetimeIndex, embargo_bars: int
) -> pd.Series:
    """Drop train observations within ``embargo_bars`` after each test block."""
    if embargo_bars <= 0 or train.empty:
        return train
    pos = {ts: i for i, ts in enumerate(bar_index)}
    n = len(bar_index)
    # Mark embargoed positions (the window strictly after each test label end)
    # once on a bitmap — O(test·embargo + train) instead of O(train·span·test).
    embargoed = np.zeros(n, dtype=bool)
    # Iterate the Series directly: this preserves tz-aware Timestamps, whereas
    # `.values` would coerce to tz-naive numpy.datetime64 and miss the dict.
    for t1_i in test_t1:
        end_pos = pos.get(t1_i)
        if end_pos is None:
            continue
        lo, hi = end_pos + 1, min(end_pos + embargo_bars, n - 1)
        if lo <= hi:
            embargoed[lo : hi + 1] = True
    keep = np.fromiter(
        (not embargoed[pos[ts]] for ts in train.index), dtype=bool, count=len(train)
    )
    return train[keep]


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


class PurgedKFold:
    """Sequential purged k-fold (AFML 7.3) — each sample in exactly one test fold.

    Unlike CPCV (which tests overlapping group combinations), this gives a clean
    partition, so it is the tool for generating **out-of-fold predictions** — the
    exact input the Layer 5 meta-model must train on. Test folds are contiguous;
    train is everything else with purge + embargo around each test block.
    """

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(self, t1: pd.Series) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_pos, test_pos)`` with contiguous, non-overlapping tests."""
        n = len(t1)
        bar_index = t1.index
        embargo_bars = int(n * self.embargo_pct)
        pos_of = {ts: i for i, ts in enumerate(bar_index)}

        for test_pos in np.array_split(np.arange(n), self.n_splits):
            test_t1 = t1.iloc[test_pos]
            surviving = purge_train_times(t1, test_t1)
            surviving = apply_embargo(surviving, test_t1, bar_index, embargo_bars)
            train_pos = np.array(sorted(pos_of[ts] for ts in surviving.index), dtype=int)
            yield train_pos, np.sort(test_pos)
