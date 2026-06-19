"""L0 — pull bars (and optionally ticks) from MT5 into TimescaleDB.

    python -m scripts.ingest --pairs EURUSD GBPUSD --timeframe H4 \
        --start 2018-01-01 --ticks

Requires the ``[mt5]`` (broker) and ``[db]`` extras and a configured ``.env``.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime

from aurax.config import get_settings, load_params
from aurax.enums import Timeframe
from aurax.l0_data import Ingestor
from aurax.logging import configure_logging, get_logger

log = get_logger("scripts.ingest")


def _parse_date(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


def main() -> None:
    params = load_params()
    default_pairs = list(params.get("instruments", {}).keys()) or ["EURUSD", "GBPUSD"]

    ap = argparse.ArgumentParser(description="Aura-X L0 ingestion")
    ap.add_argument("--pairs", nargs="+", default=default_pairs)
    ap.add_argument("--timeframe", choices=[t.value for t in Timeframe], default="H4")
    ap.add_argument("--start", type=_parse_date, default=_parse_date("2018-01-01"))
    ap.add_argument("--end", type=_parse_date, default=datetime.now(UTC))
    ap.add_argument("--ticks", action="store_true", help="also ingest tick/spread history")
    args = ap.parse_args()

    configure_logging(get_settings().log_level)
    summary = Ingestor().ingest_universe(
        pairs=args.pairs,
        timeframes=[Timeframe(args.timeframe)],
        start=args.start,
        end=args.end,
        with_ticks=args.ticks,
    )
    log.info("ingest_complete", summary=summary)
    for key, count in summary.items():
        print(f"  {key:<18} {count:>10,} rows")


if __name__ == "__main__":
    main()
