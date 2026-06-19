"""L1 — build the point-in-time feature matrix and upsert it to the store.

    python -m scripts.build_features --symbol EURUSD --timeframe H4

Requires the ``[db]`` extra. Cross-pair features are wired automatically when
the configured partner leg has stored bars.
"""

from __future__ import annotations

import argparse

from aurax.config import get_settings, load_params
from aurax.enums import Timeframe
from aurax.l1_features import build_feature_matrix
from aurax.logging import configure_logging, get_logger

log = get_logger("scripts.build_features")


def main() -> None:
    params = load_params()
    ap = argparse.ArgumentParser(description="Aura-X L1 feature build")
    ap.add_argument("--symbol", default=params.get("cross_pair", {}).get("leg_a", "EURUSD"))
    ap.add_argument("--timeframe", choices=[t.value for t in Timeframe], default="H4")
    ap.add_argument("--feature-set", default="v1")
    args = ap.parse_args()

    configure_logging(get_settings().log_level)
    from aurax.db import BarRepository, FeatureRepository

    tf = Timeframe(args.timeframe)
    repo = BarRepository()
    bars = repo.load_bars(args.symbol, tf)
    if bars.empty:
        log.warning("no_bars", symbol=args.symbol, timeframe=tf.value)
        return

    xp = params.get("cross_pair", {})
    leg_a, leg_b = xp.get("leg_a"), xp.get("leg_b")
    partner_symbol = leg_b if args.symbol == leg_a else leg_a if args.symbol == leg_b else None
    partner_close = None
    if partner_symbol:
        partner_bars = repo.load_bars(partner_symbol, tf)
        partner_close = None if partner_bars.empty else partner_bars["close"]

    matrix = build_feature_matrix(bars, params, partner_close=partner_close, dropna_warmup=True)
    n = FeatureRepository().upsert_features(matrix, args.symbol, tf, args.feature_set)
    log.info("features_built", symbol=args.symbol, rows=n, features=matrix.shape[1])
    print(f"  {args.symbol} {tf.value}: {n:,} rows × {matrix.shape[1]} features → feature_set={args.feature_set}")


if __name__ == "__main__":
    main()
