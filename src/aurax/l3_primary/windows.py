"""Sliding-window construction for sequence (deep) members.

Sequence models consume a lookback window per event. The window ending at bar
``t`` uses only bars ``<= t`` (point-in-time), so labels/predictions stay aligned
with the rest of the pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_sequences(X: pd.DataFrame, lookback: int) -> tuple[np.ndarray, pd.Index]:
    """Build ``(N, lookback, n_features)`` trailing windows.

    Returns the window tensor and the index of each window's **last** bar, so
    ``sequences[k]`` describes (and is aligned to) ``index[k]`` using the bars
    ``[k-lookback+1 .. k]``. The first ``lookback-1`` bars have no full window.
    """
    arr = np.nan_to_num(X.to_numpy(dtype=float))
    n, f = arr.shape
    if n < lookback:
        return np.empty((0, lookback, f), dtype=float), X.index[:0]
    # (n-lookback+1, f, lookback) → (n-lookback+1, lookback, f)
    windows = np.lib.stride_tricks.sliding_window_view(arr, lookback, axis=0)
    sequences = np.ascontiguousarray(windows.transpose(0, 2, 1))
    return sequences, X.index[lookback - 1 :]
