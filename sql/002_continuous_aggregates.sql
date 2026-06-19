-- ─────────────────────────────────────────────────────────────────────────────
-- Aura-X · Layer 0 storage · 002 — continuous aggregates for fast feature windows
-- These give L1 cheap rolling context without rescanning raw chunks every time.
-- ─────────────────────────────────────────────────────────────────────────────

-- Daily OHLCV rolled up from H4 bars → regime-context features (L1/L2).
CREATE MATERIALIZED VIEW IF NOT EXISTS market.bars_d1_from_h4
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    time_bucket(INTERVAL '1 day', ts)        AS day,
    first(open, ts)                          AS open,
    max(high)                                AS high,
    min(low)                                 AS low,
    last(close, ts)                          AS close,
    sum(volume)                              AS volume,
    avg(spread)                              AS avg_spread
FROM market.bars
WHERE timeframe = 'H4'
GROUP BY symbol, day
WITH NO DATA;

SELECT add_continuous_aggregate_policy('market.bars_d1_from_h4',
    start_offset      => INTERVAL '30 days',
    end_offset        => INTERVAL '1 day',
    schedule_interval => INTERVAL '4 hours',
    if_not_exists     => TRUE);

-- Hourly spread summary from ticks → realistic cost modeling / execution guards.
CREATE MATERIALIZED VIEW IF NOT EXISTS market.spread_hourly
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    time_bucket(INTERVAL '1 hour', ts)       AS hour,
    avg(spread)                              AS mean_spread,
    percentile_agg(spread)                   AS spread_pctile,  -- approx percentiles
    count(*)                                 AS n_ticks
FROM market.ticks
GROUP BY symbol, hour
WITH NO DATA;

SELECT add_continuous_aggregate_policy('market.spread_hourly',
    start_offset      => INTERVAL '7 days',
    end_offset        => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists     => TRUE);
