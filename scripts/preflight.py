"""Data-readiness preflight ("doctor") — run BEFORE fetching real data.

    python -m scripts.preflight

Checks, without ever crashing, what the data pipeline needs and prints a clear
go/no-go for each mode:

* **Live MT5 → store** — the real source (MetaTrader5 is Windows-only).
* **DB-optional parquet** — read/write ``data/*.parquet`` with no database.

Every probe is guarded; a missing optional dependency or an unreachable DB is
reported, not raised. Secrets are redacted.
"""

from __future__ import annotations

import importlib.util
import platform
import sys

from aurax.config import PROJECT_ROOT, get_instrument_specs, get_settings, load_params
from aurax.enums import Timeframe
from aurax.l0_data import MT5Client, ParquetBarStore

OK, WARN, FAIL, SKIP = "[ OK ]", "[WARN]", "[FAIL]", "[ -- ]"


def _line(mark: str, text: str) -> None:
    print(f"  {mark} {text}")


def _section(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


def _redact(value: str) -> str:
    return "***set***" if value else "(unset)"


# --- checks ------------------------------------------------------------------
def check_config() -> list[str]:
    _section("Configuration (config/*.yaml)")
    notes: list[str] = []
    try:
        params = load_params()
        specs = get_instrument_specs()
        instruments = list(specs)
        primary = params.get("timeframes", {}).get("primary", "H4")
        _line(OK, f"config loads · {len(instruments)} instrument(s): {', '.join(instruments)}")
        _line(OK, f"primary timeframe: {primary} · cross-pair: {params.get('cross_pair', {})}")
        if not instruments:
            notes.append("no instruments configured")
    except Exception as exc:  # noqa: BLE001
        _line(FAIL, f"config failed to load: {exc}")
        notes.append("config")
    return notes


def check_env() -> None:
    _section("Environment & secrets (.env)")
    env_file = PROJECT_ROOT / ".env"
    _line(OK if env_file.exists() else WARN,
          f".env present: {env_file.exists()}  ({env_file})")
    s = get_settings()
    _line(OK, f"AURAX_ENV={s.env.value} · log_level={s.log_level}")
    _line(OK, f"MT5: login={'set' if s.mt5.login else '(unset)'} · "
              f"server={s.mt5.server or '(unset)'} · password={_redact(s.mt5.password)}")
    _line(OK, f"DB:  {s.db.user}@{s.db.host}:{s.db.port}/{s.db.name} · "
              f"password={_redact(s.db.password)}")
    _line(OK, f"Telegram: {'configured' if s.telegram.bot_token else '(unset)'}")


def check_mt5() -> bool:
    _section("MT5 source (live — the real data feed)")
    client = MT5Client()
    if not client.available:
        _line(SKIP, f"MetaTrader5 package not importable on this platform ({platform.system()}).")
        _line(SKIP, "Expected: run ingestion on a Windows host with  pip install -e '.[mt5]'.")
        return False
    _line(OK, "MetaTrader5 package importable.")
    try:
        client.connect()
        _line(OK, "mt5.initialize() succeeded — terminal reachable.")
        client.shutdown()
        return True
    except Exception as exc:  # noqa: BLE001
        _line(WARN, f"package present but connect failed: {exc}")
        return False


def check_db() -> bool:
    _section("TimescaleDB sink (production storage)")
    if importlib.util.find_spec("sqlalchemy") is None:
        _line(SKIP, "sqlalchemy not installed — install the [db] extra to use the DB sink.")
        return False
    try:
        from sqlalchemy import text

        from aurax.db import get_engine

        engine = get_engine()
        with engine.connect() as conn:
            _line(OK, "database reachable.")
            for tbl in ("market.bars", "market.features", "market.labels", "market.trades"):
                exists = conn.execute(text("SELECT to_regclass(:t)"), {"t": tbl}).scalar()
                if exists is None:
                    _line(WARN, f"table {tbl} missing — apply sql/ migrations (Timescale).")
                else:
                    n = conn.execute(text(f"SELECT count(*) FROM {tbl}")).scalar()
                    _line(OK, f"table {tbl}: {n:,} rows")
        return True
    except Exception as exc:  # noqa: BLE001
        _line(SKIP, f"database not reachable ({type(exc).__name__}) — fine for parquet mode.")
        return False


def check_parquet() -> bool:
    _section("DB-optional parquet store (data/*.parquet)")
    has_engine = importlib.util.find_spec("pyarrow") is not None or (
        importlib.util.find_spec("fastparquet") is not None
    )
    if not has_engine:
        _line(FAIL, "no parquet engine — install with  pip install -e '.[files]'  (pyarrow).")
        return False
    _line(OK, "parquet engine (pyarrow) available.")

    root = PROJECT_ROOT / "data"
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / ".preflight_write_test"
        probe.write_text("ok")
        probe.unlink()
        _line(OK, f"data root writable: {root}")
    except Exception as exc:  # noqa: BLE001
        _line(FAIL, f"data root not writable: {exc}")
        return False

    store = ParquetBarStore(root)
    try:
        params = load_params()
        primary = Timeframe(params.get("timeframes", {}).get("primary", "H4"))
        instruments = list(get_instrument_specs())
    except Exception:  # noqa: BLE001
        instruments, primary = [], Timeframe.H4
    found = False
    for sym in instruments:
        n, first, last = store.coverage(sym, primary)
        if n:
            found = True
            _line(OK, f"bars on disk · {sym} {primary.value}: {n:,} rows  {first} … {last}")
        else:
            _line(SKIP, f"no bars yet · {sym} {primary.value}")
    if not found:
        _line(WARN, "no bars stored yet — run:  python -m scripts.ingest --dest parquet  (on a Windows MT5 host)")
    return True


def main() -> None:
    print("Aura-X · data-readiness preflight")
    print("=" * 33)
    check_config()
    check_env()
    mt5_ready = check_mt5()
    db_ready = check_db()
    parquet_ready = check_parquet()

    _section("Readiness verdict")
    if parquet_ready:
        _line(OK, "DB-optional parquet pipeline: READY — features/labels/validate can run here on parquet bars.")
    else:
        _line(FAIL, "DB-optional parquet pipeline: install the [files] extra (pyarrow).")

    if mt5_ready and (db_ready or parquet_ready):
        _line(OK, "Live ingest: READY — MT5 reachable and a sink is available.")
    else:
        missing = []
        if not mt5_ready:
            missing.append("MT5 (run on a Windows host with the [mt5] extra)")
        if not (db_ready or parquet_ready):
            missing.append("a sink (DB or parquet)")
        _line(WARN, "Live ingest: not runnable here — needs " + " + ".join(missing) + ".")

    print()


if __name__ == "__main__":
    sys.exit(main())
