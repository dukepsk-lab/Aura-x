"""L1 — build the point-in-time feature matrix and upsert it to a store.

    # from TimescaleDB (needs [db])
    python -m scripts.build_features --symbol EURUSD --timeframe H4

    # DB-optional: read parquet bars produced by `ingest --dest parquet`, write parquet features
    python -m scripts.build_features --symbol EURUSD --timeframe H4 --source parquet --dest parquet

Cross-pair features are wired automatically when the configured partner leg has
stored bars in the same source.
"""

from __future__ import annotations

import argparse

from aurax.config import get_settings, load_params
from aurax.enums import Timeframe
from aurax.l0_data import make_bar_store, make_feature_store
from aurax.l1_features import build_feature_matrix
from aurax.logging import configure_logging, get_logger

log = get_logger("scripts.build_features")


def main() -> None:
    params = load_params()
    ap = argparse.ArgumentParser(description="Aura-X L1 feature build")
    ap.add_argument("--symbol", default=params.get("cross_pair", {}).get("leg_a", "EURUSD"))
    ap.add_argument("--timeframe", choices=[t.value for t in Timeframe], default="H4")
    ap.add_argument("--feature-set", default="v1")
    ap.add_argument("--source", choices=["db", "parquet"], default="db", help="bar source")
    ap.add_argument("--dest", choices=["db", "parquet"], default="db", help="feature sink")
    ap.add_argument("--data-root", default=None, help="parquet root (default: ./data)")
    args = ap.parse_args()

    configure_logging(get_settings().log_level)

    tf = Timeframe(args.timeframe)
    bar_store = make_bar_store(args.source, args.data_root)
    bars = bar_store.load_bars(args.symbol, tf)
    if bars.empty:
        log.warning("no_bars", symbol=args.symbol, timeframe=tf.value, source=args.source)
        return

    xp = params.get("cross_pair", {})
    leg_a, leg_b = xp.get("leg_a"), xp.get("leg_b")
    partner_symbol = leg_b if args.symbol == leg_a else leg_a if args.symbol == leg_b else None
    partner_close = None
    if partner_symbol:
        partner_bars = bar_store.load_bars(partner_symbol, tf)
        partner_close = None if partner_bars.empty else partner_bars["close"]

    matrix = build_feature_matrix(bars, params, partner_close=partner_close, dropna_warmup=True)
    feature_store = make_feature_store(args.dest, args.data_root)
    n = feature_store.upsert_features(matrix, args.symbol, tf, args.feature_set)
    log.info("features_built", symbol=args.symbol, rows=n, features=matrix.shape[1], dest=args.dest)
    print(f"  {args.symbol} {tf.value}: {n:,} rows × {matrix.shape[1]} features → feature_set={args.feature_set} ({args.dest})")


if __name__ == "__main__":
    main()
