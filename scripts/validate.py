"""Run the §5 validation gate end-to-end (L1 features → L4 labels → CPCV/WF/holdout).

    python -m scripts.validate --demo                 # synthetic, no DB/MT5 needed
    python -m scripts.validate --symbol EURUSD        # from TimescaleDB ([db] extra)

Until the Layer-3 ensemble lands, a small **linear sign model** stands in as the
primary so the gatekeeper is runnable now. On random-walk demo data the correct,
expected verdict is **NO-GO** — that is the gate doing its job (rejecting noise).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from aurax.config import get_settings, load_params
from aurax.enums import Timeframe
from aurax.l1_features import build_feature_matrix
from aurax.l4_labeling import LabelConfig, Labeler
from aurax.logging import configure_logging, get_logger
from aurax.validation import ValidationConfig, run_validation

log = get_logger("scripts.validate")


class LinearSignModel:
    """Placeholder primary (stand-in for L3): weighted least-squares → sign.

    Predicts direction as ``sign(Xβ)`` with β from a (sample-weighted) linear fit
    of the label on the features. Replace with the L3 ensemble once built.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None = None):
        a = np.c_[np.ones(len(X)), np.nan_to_num(X.to_numpy(dtype=float))]
        target = y.to_numpy(dtype=float)
        if sample_weight is not None:
            w = np.sqrt(np.clip(sample_weight.to_numpy(dtype=float), 0, None))[:, None]
            a, target = a * w, target * w[:, 0]
        self.coef_, *_ = np.linalg.lstsq(a, target, rcond=None)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        a = np.c_[np.ones(len(X)), np.nan_to_num(X.to_numpy(dtype=float))]
        return np.sign(a @ self.coef_)


def _demo_bars(n: int = 2400, seed: int = 7) -> tuple[pd.DataFrame, pd.Series]:
    idx = pd.date_range("2019-01-01", periods=n, freq="4h", tz="UTC")
    rng = np.random.default_rng(seed)

    def make(level, vol, s):
        c = level + np.cumsum(np.random.default_rng(s).normal(0, vol, n))
        o = np.r_[c[0], c[:-1]]
        wick = rng.uniform(2e-4, 1e-3, n)
        return pd.DataFrame(
            {"open": o, "high": np.maximum(o, c) + wick,
             "low": np.minimum(o, c) - wick, "close": c,
             "volume": rng.uniform(800, 1200, n)}, index=idx)

    return make(1.10, 0.0015, 1), make(1.30, 0.0018, 2)["close"]


def _load_bars(symbol: str, tf: Timeframe, params: dict):
    from aurax.db import BarRepository

    repo = BarRepository()
    bars = repo.load_bars(symbol, tf)
    xp = params.get("cross_pair", {})
    leg_a, leg_b = xp.get("leg_a"), xp.get("leg_b")
    partner = leg_b if symbol == leg_a else leg_a if symbol == leg_b else None
    partner_close = repo.load_bars(partner, tf)["close"] if partner else None
    return bars, partner_close


def main() -> None:
    params = load_params()
    ap = argparse.ArgumentParser(description="Aura-X validation gate")
    ap.add_argument("--demo", action="store_true", help="use synthetic data (no DB)")
    ap.add_argument("--symbol", default=params.get("cross_pair", {}).get("leg_a", "EURUSD"))
    ap.add_argument("--timeframe", choices=[t.value for t in Timeframe], default="H4")
    ap.add_argument("--n-trials", type=int, default=1, help="# configurations tried (be honest)")
    args = ap.parse_args()
    configure_logging(get_settings().log_level)

    tf = Timeframe(args.timeframe)
    bars, partner_close = _demo_bars() if args.demo else _load_bars(args.symbol, tf, params)
    if bars is None or bars.empty:
        log.warning("no_bars", symbol=args.symbol)
        return

    # L1 features + L4 labels, aligned on usable (non-warmup) events.
    features = build_feature_matrix(bars, params, partner_close=partner_close).dropna()
    labels = Labeler(LabelConfig.from_params(params)).make_labels(bars)
    common = labels.index.intersection(features.index)
    if len(common) < 200:
        log.warning("insufficient_aligned_rows", n=len(common))
        return
    X = features.loc[common]
    lab = labels.loc[common]

    cfg = ValidationConfig.from_params(params)
    cfg.n_trials = args.n_trials
    report = run_validation(
        LinearSignModel,
        X,
        lab["label"].astype(float),
        t1=lab["t1"],
        ret=lab["ret"],
        labels=lab["label"].astype(float),
        close=bars["close"].reindex(common),
        sample_weight=lab["sample_weight"],
        config=cfg,
    )

    src = "demo synthetic" if args.demo else f"{args.symbol} {tf.value}"
    print(f"\n── Aura-X validation gate · {src} · {len(common):,} events ──")
    print(report.summary())
    if args.demo and not report.passed:
        print("\n(NO-GO on random-walk demo data is expected — the gate is rejecting noise.)")


if __name__ == "__main__":
    main()
