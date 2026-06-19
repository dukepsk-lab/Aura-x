"""Meta-feature construction for the TRUST model.

The meta-model sees the **primary's signal plus context** — never the raw price
features alone — because its job is to judge *the primary*, not re-predict
direction:

* **primary confidence** — directional margin + class probabilities;
* **regime** (one-hot from Layer 2) and **volatility** (ATR-pct / realized vol);
* **recent performance** — trailing win-rate of the primary's recent bets
  (lagged, point-in-time), capturing edge decay / persistence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..enums import Regime

_REGIME_DUMMIES = [f"regime_{r.value}" for r in Regime]


def build_meta_features(
    primary_proba: pd.DataFrame,
    primary_side: pd.Series,
    *,
    regimes: pd.Series | None = None,
    context: pd.DataFrame | None = None,
    meta_labels: pd.Series | None = None,
    recent_window: int = 20,
) -> pd.DataFrame:
    """Assemble the meta-feature matrix aligned to ``primary_side.index``.

    ``primary_proba`` has L3's ``p_down/p_flat/p_up`` columns; ``context`` is an
    optional frame of L1 volatility features to pass through; ``meta_labels`` (if
    given) seeds the lagged recent-win-rate feature.
    """
    idx = primary_side.index
    out = pd.DataFrame(index=idx)
    out["prim_margin"] = (primary_proba["p_up"] - primary_proba["p_down"]).abs()
    out["prim_conf"] = primary_proba[["p_down", "p_flat", "p_up"]].max(axis=1)
    out["prim_p_up"] = primary_proba["p_up"]
    out["prim_p_down"] = primary_proba["p_down"]
    out["prim_side"] = primary_side.astype(float)

    if regimes is not None:
        reg = regimes.reindex(idx).astype(object)
        for r in Regime:
            out[f"regime_{r.value}"] = (reg == r.value).astype(float)

    if context is not None:
        for col in context.columns:
            out[f"ctx_{col}"] = context[col].reindex(idx)

    if meta_labels is not None:
        # Trailing win-rate of the primary's PRIOR bets (shift → point-in-time).
        ml = meta_labels.reindex(idx).astype(float)
        out["prim_recent_winrate"] = (
            ml.shift(1).rolling(recent_window, min_periods=max(3, recent_window // 4)).mean()
        )
        out["prim_recent_winrate"] = out["prim_recent_winrate"].fillna(0.5)

    return out.replace([np.inf, -np.inf], np.nan)
