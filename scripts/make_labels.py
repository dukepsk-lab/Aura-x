"""L4 — generate triple-barrier labels + uniqueness weights and upsert them.

    python -m scripts.make_labels --symbol EURUSD --timeframe H4

Training-time only. Requires the ``[db]`` extra.
"""

from __future__ import annotations

import argparse

from aurax.config import get_settings, load_params
from aurax.enums import Timeframe
from aurax.l4_labeling import LabelConfig, Labeler
from aurax.logging import configure_logging, get_logger

log = get_logger("scripts.make_labels")


def main() -> None:
    params = load_params()
    ap = argparse.ArgumentParser(description="Aura-X L4 labeling")
    ap.add_argument("--symbol", default=params.get("cross_pair", {}).get("leg_a", "EURUSD"))
    ap.add_argument("--timeframe", choices=[t.value for t in Timeframe], default="H4")
    ap.add_argument("--label-set", default="tb_v1")
    args = ap.parse_args()

    configure_logging(get_settings().log_level)
    from aurax.db import BarRepository, LabelRepository

    tf = Timeframe(args.timeframe)
    bars = BarRepository().load_bars(args.symbol, tf)
    if bars.empty:
        log.warning("no_bars", symbol=args.symbol, timeframe=tf.value)
        return

    labels = Labeler(LabelConfig.from_params(params)).make_labels(bars)
    n = LabelRepository().upsert_labels(labels, args.symbol, tf, args.label_set)
    dist = labels["label"].value_counts().sort_index().to_dict()
    log.info("labels_built", symbol=args.symbol, rows=n, distribution=dist)
    print(f"  {args.symbol} {tf.value}: {n:,} labels → label_set={args.label_set}")
    print(f"  distribution (-1/0/+1): {dist}")


if __name__ == "__main__":
    main()
