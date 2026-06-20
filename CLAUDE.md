# CLAUDE.md

Guidance for Claude Code (and future contributors) working in this repository.

## What this is

Aura-X is a layered, defensive ML trading system for EURUSD/GBPUSD on H4 via
MT5. The design philosophy (meta-labeling): a primary model is a noisy idea
generator; a second model decides whether each idea is worth risking capital
on; volatility decides how much; a regime layer decides which behavior is even
appropriate right now. `docs/architecture.md` is the source of truth for the
design rationale and must not be treated as stale — if code and doc disagree,
read both before changing either.

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # data/feature/labeling stack only
pip install -e ".[all]"          # db + api + models + monitoring + dev
pip install -e ".[models]"       # + scikit-learn/lightgbm/catboost/hmmlearn/statsmodels
pip install -e ".[deep]"         # + torch (CNN/PatchTST/SSM ensemble members)
pip install -e ".[mt5]"          # + MetaTrader5 (Windows-only)

make test                        # pytest, full suite
make cov                         # pytest --cov=aurax --cov-report=term-missing
make lint                        # ruff check src tests scripts
make fmt                         # ruff format
make typecheck                   # mypy src
python -m scripts.validate --demo   # run the validation gate on synthetic data (no DB/MT5)

make doctor                      # scripts.preflight — data-pipeline readiness (MT5/DB/parquet)
make db-up / make db-down        # TimescaleDB via docker compose (migrations in ./sql auto-run)
make api                         # uvicorn aurax.api.main:app --reload
```

Fetching real data: MT5 is the **source** (Windows-only pull); the **sink** is
either TimescaleDB or DB-optional local **parquet** (`l0_data.store`, the
`[files]` extra). `scripts.{ingest,build_features,make_labels}` take
`--dest/--source {db,parquet}` so the L1→L4→validation path runs with no
database. See `docs/data_ingestion.md`.

Run a single test file/test the normal pytest way, e.g.
`pytest tests/test_l6_risk.py -k correlation_cap -v`.

Always run `ruff check src tests scripts` and the full `pytest` suite before
considering a change done — both must be clean, not just the file you touched.

## Architecture: the 8-layer pipeline + validation + lifecycle

| Layer | Module | Role |
|------:|--------|------|
| L0 | `aurax.l0_data` | MT5 → TimescaleDB ingestion (OHLCV H4/D1/M15 + tick/spread) |
| L1 | `aurax.l1_features` | ATR · Yang-Zhang · Hurst · KER · cross-pair · session features |
| L2 | `aurax.l2_regime` | Gaussian HMM + Hurst/KER gate + shock stand-down (trend/range/shock) |
| L3 | `aurax.l3_primary` | Primary SIDE model — GBM/logistic + deep (CNN/PatchTST/SSM) ensemble, regime-conditional |
| L4 | `aurax.l4_labeling` | Triple-Barrier (ATR-scaled) + sample-uniqueness weights + trend-scanning — **training only** |
| L5 | `aurax.l5_meta` | Meta-label TRUST model — calibrated meta-learner on OOF predictions, P(correct) gate at τ |
| L6 | `aurax.l6_risk` | `RiskManager` — ATR sizing × confidence × covariance correlation cap + circuit breaker |
| L6b | `aurax.l6_risk.allocation` | Dirichlet-policy PPO allocator across the instrument simplex |
| L7 | `aurax.l7_execution` | Guarded orders (spread/news/slippage), idempotent IDs, retry, ATR stops · Paper/MT5 brokers |
| L8 | `aurax.l8_monitoring` | `Monitor` — fill/breach/regime/decay alerts, rolling Sharpe + calibration drift |
| — | `aurax.validation` | CPCV (purge+embargo), purged-KFold OOF, walk-forward, holdout, baselines, deflated Sharpe, Go/No-Go |
| — | `aurax.lifecycle` | `TrainingPipeline` (fit+persist) and `RetrainOrchestrator` (decay → retrain → gate → promote) |

**Training path:** `L0 → L1 → L4 (labels+weights) → L2/L3 (fit primary) → L5 (fit meta on out-of-fold preds) → validation`
**Inference path:** `L0 → L1 → L2 → L3 → L5 → L6 → L6b → L7 → L8`

These paths are kept strictly separate in code. Triple-Barrier and CPCV exist
*only* in training. The meta-model (L5) must always be trained on the
primary's **out-of-fold** predictions (`aurax.validation.oof_predict` /
`aurax.l5_meta.train_meta_labeler`) — never in-sample predictions, that would
leak and silently inflate the meta-model's apparent skill.

Nothing reaches L7 without clearing `aurax.validation.run_validation`. When
adding or changing a model, the question is never "does it look good on one
split" — it's whether `run_validation(...).passed` is `True` after CPCV,
walk-forward, holdout-vs-baselines, and deflated Sharpe. `lifecycle.RetrainOrchestrator`
is the automated version of this same discipline: it only promotes a
retrained model if it clears the gate.

## Conventions that matter (violating these reintroduces leakage)

- **Point-in-time discipline.** Every feature/label may only use data
  available at or before its own timestamp. Rolling windows are trailing.
  Never use `.shift(-k)` outside `l4_labeling` (the one place looking forward
  is the point).
- **sklearn-style estimator protocol** (`fit` / `predict` / `predict_proba`)
  is used uniformly across L2/L3/L5 members so any of them drops straight
  into `run_validation` / `oof_predict` without adapters.
- **Heavy deps are optional and lazy-imported**, with a dependency-free
  fallback so the core pipeline (and its tests) run with nothing but
  numpy/pandas/scipy installed:
  - L3: `LogisticSide` (numpy) vs. `LightGBMSide`/`CatBoostSide`/`CNNSide`/`PatchTSTSide`/`SSMSide` (lazy `lightgbm`/`catboost`/`torch`)
  - L2: numpy/scipy `GaussianHMM` (no `hmmlearn` requirement)
  - L7: `PaperBroker` vs. MT5 broker
  - L8: log-based notifier vs. Telegram
  Don't hardcode a heavy import at module top level outside these guarded
  paths — it breaks the "works with `pip install -e .` alone" promise.
  `available_backends()` (L3) reports what's actually importable.
  - **LightGBM/CatBoost classes/objective are inferred from `y`, never
  hardcoded** (`objective="multiclass"` blows up if a CV fold only contains 2
  of the 3 classes — this bit us once already, see git history on `l3_primary/members.py`).
- **Config layering:** secrets/env-specific values via `pydantic-settings`
  (`.env`, see `.env.example`); versioned algorithm parameters via YAML in
  `config/*.yaml`, merged by `aurax.config.load_params()`. Don't put tunable
  thresholds in code when they belong in `config/default.yaml`.
- **Persistence:** `lifecycle.TrainingPipeline` pickles the fitted bundle.
  Anything reachable from a `TrainedModel` must stay picklable — no
  `lambda`/local-closure factories on persisted objects (see
  `l5_meta/meta_model.py`'s module-level `_default_meta_base`, added
  specifically to fix this).

## Testing notes

- The CPCV purge/embargo path is the single most leakage-sensitive piece of
  code in the repo and has already hidden one critical bug silently
  (uninitialized-memory interval merge that purged entire folds to zero rows
  without raising). Any change to `aurax.validation.cpcv` needs the
  `len(train_pos) > 0` regression-style assertions kept intact, not loosened.
- Deep (torch) member tests in `tests/test_l3_deep.py` skip cleanly via an
  `importlib.util.find_spec("torch")` guard — don't make them hard-require
  torch.
- `python -m scripts.validate --demo` on pure noise must report **NO-GO**;
  if a harness change makes it report GO on noise, that's the bug, not a
  feature.
- Tests must avoid degenerate statistics (constant/zero-variance return
  series, near-50/50 targets) — several earlier test failures were the
  test's fault (an undefined-Sharpe constant series, a coin-flip-level
  signal) rather than the implementation's.

## Git

- Always check the current branch before committing; this repo is typically
  worked on a feature branch rather than `main`.
- Commit messages: short imperative summary line, blank line, then the
  rationale/what-changed in prose. Don't bundle unrelated layers into one
  commit — each layer/feature lands as its own commit.
- Run lint + full test suite before every commit. Don't commit with failing
  tests or ruff errors.
