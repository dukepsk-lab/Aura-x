"""Run the §5 validation gate end-to-end (L1 features → L4 labels → CPCV/WF/holdout).

    python -m scripts.validate --demo                 # synthetic, no DB/MT5 needed
    python -m scripts.validate --symbol EURUSD        # from TimescaleDB ([db] extra)

The primary is the real **L3 ensemble** (LightGBM/CatBoost if ``[models]`` is
installed, else the dependency-free logistic member). On random-walk demo data
the correct, expected verdict is **NO-GO** — the gate doing its job (rejecting
noise); a genuine edge would clear CPCV + walk-forward + holdout + deflation.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from aurax.config import get_settings, load_params
from aurax.enums import Timeframe
from aurax.l1_features import build_feature_matrix
from aurax.l2_regime import RegimeConfig, RegimeDetector
from aurax.l3_primary import PrimaryConfig, PrimarySignalModel, available_backends
from aurax.l4_labeling import LabelConfig, Labeler
from aurax.l5_meta import MetaConfig, MetaGatedPrimary
from aurax.logging import configure_logging, get_logger
from aurax.validation import ValidationConfig, run_validation

log = get_logger("scripts.validate")


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
    ap.add_argument("--meta", action="store_true", help="gate the primary with the L5 meta-model")
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

    # L2 regime context (fit the HMM on the same features and report the mix).
    regimes = None
    try:
        det = RegimeDetector(RegimeConfig.from_params(params)).fit(X)
        regimes = det.regime_series(X)
        mix = regimes.value_counts(normalize=True).round(2).to_dict()
    except Exception as exc:  # noqa: BLE001 - regime context is best-effort here
        log.warning("regime_fit_failed", err=str(exc))
        mix = {}

    cfg = ValidationConfig.from_params(params)
    cfg.n_trials = args.n_trials
    primary = PrimaryConfig.from_available(params)

    def make_primary():
        return PrimarySignalModel(primary)

    if args.meta:  # L3 primary gated by the L5 meta-model
        ctx = [c for c in ("vol_atr_pct", "vol_realized_vol", "tm_hurst") if c in X.columns]
        meta_cfg = MetaConfig.from_params(params)
        make_estimator = lambda: MetaGatedPrimary(  # noqa: E731
            make_primary, t1=lab["t1"], regimes=regimes, context_cols=ctx, meta_config=meta_cfg
        )
    else:
        make_estimator = make_primary

    report = run_validation(
        make_estimator,
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
    stack = "L3→L5 meta-gated" if args.meta else "L3 primary (ungated)"
    print(f"\n── Aura-X validation gate · {src} · {len(common):,} events ──")
    print(f"estimator      : {stack}  ·  members {list(primary.members)} (backends {available_backends()})")
    if mix:
        print(f"L2 regime mix  : {mix}")
    print(report.summary())
    if args.demo and not report.passed:
        print("\n(NO-GO on random-walk demo data is expected — the gate is rejecting noise.)")


if __name__ == "__main__":
    main()
