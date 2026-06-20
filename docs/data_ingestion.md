# Fetching real data — ingestion runbook

How real EURUSD/GBPUSD H4 data gets into Aura-X, and how to run the training
path on it **without a database**.

The market-data **source is MetaTrader 5**. The `MetaTrader5` Python package
ships a **Windows-only** wheel, so the *pull* must run on a Windows MT5 host.
Storage is decoupled from the pull: bars can land in **TimescaleDB** (production)
or in **local parquet** (`data/*.parquet`) so the rest of the pipeline
(L1 features → L4 labels → validation) runs anywhere — including a Linux box
with no Docker/Timescale.

```
┌─────────────────────────┐     parquet files      ┌──────────────────────────┐
│  Windows MT5 host        │  ───────────────────▶  │  Any machine (Linux/mac) │
│  scripts.ingest --dest   │   data/bars/*.parquet  │  features → labels →     │
│  parquet                 │                        │  validate  (no DB)       │
└─────────────────────────┘                         └──────────────────────────┘
```

> **Single-source-of-truth caveat.** In production the architecture wants
> TimescaleDB to back *both* training and live inference (that is what removes
> train/serve skew). The parquet path is for **research / backfills / running
> the gate on real data before any capital** — not for live trading.

---

## 0. Preflight (run this first, anywhere)

```bash
make doctor          # == python -m scripts.preflight
```

It prints a guarded go/no-go for config, `.env`, MT5, the DB sink, and the
parquet store — and never crashes on a missing dependency. On a Linux dev box
you should see **MT5 = not importable** and **parquet pipeline = READY**.

---

## 1. On the Windows MT5 host — pull bars

```powershell
python -m venv .venv ; .venv\Scripts\activate
pip install -e ".[mt5,files]"          # broker wheel + parquet writer

copy .env.example .env                  # then edit: AURAX_MT5__LOGIN / PASSWORD / SERVER
python -m scripts.preflight             # expect MT5 = reachable

# Pull H4 history for both legs straight to parquet under .\data
python -m scripts.ingest --pairs EURUSD GBPUSD --timeframe H4 --start 2015-01-01 --dest parquet
```

Outputs `data/bars/EURUSD/H4.parquet` and `data/bars/GBPUSD/H4.parquet`
(idempotent — rerunning extends/updates, never duplicates).

**Symbol names** must match your broker exactly. Some brokers suffix them
(`EURUSD.pro`, `EURUSD-m`); pass those literal names to `--pairs` and mirror
them in `config/instruments.yaml` if you want them as the configured universe.

> Tick/spread history (`--ticks`) is **DB-only** — it feeds the Timescale
> continuous aggregates. Per-bar `spread` is already captured in the H4 bars,
> which is enough for the cost-adjusted backtest. For `--dest parquet`, `--ticks`
> is skipped with a warning.

### Production variant — straight to TimescaleDB

```powershell
pip install -e ".[mt5,db]"
# point AURAX_DB__* at your Timescale instance in .env, then:
python -m scripts.ingest --pairs EURUSD GBPUSD --timeframe H4 --start 2015-01-01 --dest db --ticks
```

---

## 2. Anywhere — run the training path on the parquet bars

Copy the `data/` folder over (or just work on the Windows host). No DB needed:

```bash
pip install -e ".[files,models]"        # parquet + the L3/L5 model stack

python -m scripts.build_features --symbol EURUSD --timeframe H4 --source parquet --dest parquet
python -m scripts.build_features --symbol GBPUSD --timeframe H4 --source parquet --dest parquet
python -m scripts.make_labels    --symbol EURUSD --timeframe H4 --source parquet --dest parquet
python -m scripts.make_labels    --symbol GBPUSD --timeframe H4 --source parquet --dest parquet
```

This writes `data/features/{symbol}/H4/v1.parquet` and
`data/labels/{symbol}/H4/tb_v1.parquet`. `make doctor` then reports the stored
bar coverage (row count + date range) per instrument.

`--data-root PATH` points any script at a parquet root other than `./data`.

---

## 3. The honest next question

With real features + labels on disk, the next step is the **validation gate**
(`scripts.validate`) on real data rather than `--demo` noise — the first
real read on whether there is an edge on EURUSD/GBPUSD H4. Wiring the gate to
read the parquet feature/label stores is the natural follow-up to this runbook.

---

## Reference — paths, formats, conventions

| What | Where | Notes |
|------|-------|-------|
| Bars | `data/bars/{symbol}/{tf}.parquet` | OHLCV(+spread), index `ts` (UTC, bar **open** time) |
| Features | `data/features/{symbol}/{tf}/{set}.parquet` | wide point-in-time matrix |
| Labels | `data/labels/{symbol}/{tf}/{set}.parquet` | triple-barrier + uniqueness weights (training only) |
| Secrets | `.env` (gitignored) | `AURAX_` prefix, `__` nesting — never committed |
| Universe | `config/instruments.yaml` | symbol specs (pip size, lot step, …) |

All timestamps are UTC and point-in-time correct: a feature at `ts` uses only
bars with index `<= ts`. `data/` and `*.parquet` are gitignored — real market
data and generated artifacts never enter version control.
