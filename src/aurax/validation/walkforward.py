"""Walk-forward validation — sequential, leakage-purged, with a final holdout.

CPCV stress-tests many train/test combinations; walk-forward confirms the edge
survives *forward in time* the way live trading actually unfolds. The §5 protocol
requires both, plus a held-out, **never-touched** tail period before any capital.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

from .cpcv import purge_train_times


def holdout_split(t1: pd.Series, holdout_pct: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Carve a final never-touched holdout off the end of the timeline.

    Returns ``(dev_pos, holdout_pos)``. The dev portion is purged of any label
    that bleeds into the holdout, so the holdout stays genuinely unseen.
    """
    n = len(t1)
    cut = int(n * (1.0 - holdout_pct))
    dev_pos = np.arange(cut)
    holdout_pos = np.arange(cut, n)
    if holdout_pos.size:
        holdout_t1 = t1.iloc[holdout_pos]
        surviving = purge_train_times(t1.iloc[dev_pos], holdout_t1)
        pos_of = {ts: i for i, ts in enumerate(t1.index)}
        dev_pos = np.array(sorted(pos_of[ts] for ts in surviving.index), dtype=int)
    return dev_pos, holdout_pos


class WalkForwardSplit:
    """Sequential walk-forward folds (anchored/expanding or rolling window)."""

    def __init__(
        self, n_splits: int = 5, mode: str = "anchored", embargo_pct: float = 0.0
    ) -> None:
        if mode not in ("anchored", "rolling"):
            raise ValueError("mode must be 'anchored' or 'rolling'")
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.mode = mode
        self.embargo_pct = embargo_pct

    def split(self, t1: pd.Series) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_pos, test_pos)`` where train always precedes test.

        The boundary is purged: any train label whose span bleeds into the test
        block (plus an optional embargo gap) is dropped.
        """
        n = len(t1)
        folds = np.array_split(np.arange(n), self.n_splits + 1)
        embargo = int(n * self.embargo_pct)

        for i in range(1, self.n_splits + 1):
            test_pos = folds[i]
            if test_pos.size == 0:
                continue
            test_start = test_pos[0]
            train_end = max(0, test_start - embargo)
            train_start = 0 if self.mode == "anchored" else folds[i - 1][0]
            train_pos = np.arange(train_start, train_end)
            if train_pos.size == 0:
                continue
            # Purge train labels that overlap the test block in time.
            surviving = purge_train_times(t1.iloc[train_pos], t1.iloc[test_pos])
            pos_of = {ts: p for p, ts in enumerate(t1.index)}
            train_pos = np.array(sorted(pos_of[ts] for ts in surviving.index), dtype=int)
            yield train_pos, test_pos
