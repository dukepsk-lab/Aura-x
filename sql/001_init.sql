-- ─────────────────────────────────────────────────────────────────────────────
-- Aura-X · Layer 0 storage · 001 — schema, extensions, raw bar/tick hypertables
-- TimescaleDB is the single source of truth for BOTH training and live inference
-- (this is what eliminates train/serve skew). Runs automatically on first boot
-- via docker-entrypoint-initdb.d.
-- ─────────────────────────────────────────────────────────────────────────────

CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

CREATE SCHEMA IF NOT EXISTS market;

-- --- OHLCV bars (H4 primary, D1 regime context, M15 execution context) ---------
CREATE TABLE IF NOT EXISTS market.bars (
    symbol      TEXT             NOT NULL,
    timeframe   TEXT             NOT NULL,            -- 'H4' | 'D1' | 'M15'
    ts          TIMESTAMPTZ      NOT NULL,            -- bar OPEN time (UTC)
    open        DOUBLE PRECISION NOT NULL,
    high        DOUBLE PRECISION NOT NULL,
    low         DOUBLE PRECISION NOT NULL,
    close       DOUBLE PRECISION NOT NULL,
    volume      DOUBLE PRECISION NOT NULL DEFAULT 0,  -- tick volume
    spread      DOUBLE PRECISION,                     -- avg spread in points (cost model)
    ingested_at TIMESTAMPTZ      NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, timeframe, ts)
);

SELECT create_hypertable(
    'market.bars', 'ts',
    chunk_time_interval => INTERVAL '90 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS bars_symbol_tf_ts_idx
    ON market.bars (symbol, timeframe, ts DESC);

-- --- Tick / spread history (non-negotiable: realistic cost modeling needs it) ---
CREATE TABLE IF NOT EXISTS market.ticks (
    symbol      TEXT             NOT NULL,
    ts          TIMESTAMPTZ      NOT NULL,
    bid         DOUBLE PRECISION NOT NULL,
    ask         DOUBLE PRECISION NOT NULL,
    last        DOUBLE PRECISION,
    volume      DOUBLE PRECISION,
    spread      DOUBLE PRECISION GENERATED ALWAYS AS (ask - bid) STORED,
    PRIMARY KEY (symbol, ts)
);

SELECT create_hypertable(
    'market.ticks', 'ts',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS ticks_symbol_ts_idx
    ON market.ticks (symbol, ts DESC);

-- Compression policies keep tick/bar history cheap to retain for backtests.
ALTER TABLE market.ticks SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol'
);
SELECT add_compression_policy('market.ticks', INTERVAL '30 days', if_not_exists => TRUE);

ALTER TABLE market.bars SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol, timeframe'
);
SELECT add_compression_policy('market.bars', INTERVAL '180 days', if_not_exists => TRUE);
