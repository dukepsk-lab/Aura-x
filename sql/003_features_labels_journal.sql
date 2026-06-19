-- ─────────────────────────────────────────────────────────────────────────────
-- Aura-X · 003 — feature store (L1), label store (L4), trade journal (L7/L8)
-- Feature/label rows are versioned by *_set so retrains never silently overwrite.
-- ─────────────────────────────────────────────────────────────────────────────

-- --- Feature store (L1 output) -------------------------------------------------
-- Wide feature vectors stored as JSONB keyed by a named, versioned feature set.
CREATE TABLE IF NOT EXISTS market.features (
    symbol       TEXT        NOT NULL,
    timeframe    TEXT        NOT NULL,
    ts           TIMESTAMPTZ NOT NULL,        -- bar close the features describe
    feature_set  TEXT        NOT NULL,        -- e.g. 'v1'
    features     JSONB       NOT NULL,
    computed_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, timeframe, feature_set, ts)
);
SELECT create_hypertable('market.features', 'ts',
    chunk_time_interval => INTERVAL '180 days', if_not_exists => TRUE);

-- --- Label store (L4 output, TRAINING ONLY) ------------------------------------
-- Triple-barrier outcomes + sample-uniqueness weights.
CREATE TABLE IF NOT EXISTS market.labels (
    symbol        TEXT             NOT NULL,
    timeframe     TEXT             NOT NULL,
    ts            TIMESTAMPTZ      NOT NULL,    -- event start (bar close)
    label_set     TEXT             NOT NULL,    -- e.g. 'tb_v1'
    t1            TIMESTAMPTZ,                  -- first barrier touch / vertical
    label         SMALLINT         NOT NULL,    -- -1 short | 0 neutral | +1 long
    ret           DOUBLE PRECISION,            -- realized return to t1
    barrier       TEXT,                        -- 'tp' | 'sl' | 'vertical'
    tp_price      DOUBLE PRECISION,
    sl_price      DOUBLE PRECISION,
    sample_weight DOUBLE PRECISION NOT NULL DEFAULT 1.0,  -- uniqueness * decay
    side          SMALLINT,                    -- primary side if meta-labeling
    created_at    TIMESTAMPTZ      NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, timeframe, label_set, ts)
);
SELECT create_hypertable('market.labels', 'ts',
    chunk_time_interval => INTERVAL '180 days', if_not_exists => TRUE);

-- --- Trade journal (L7 fills → L8 monitoring → CPCV refresh) --------------------
CREATE TABLE IF NOT EXISTS market.trades (
    trade_id      TEXT             PRIMARY KEY,  -- idempotent client order id
    symbol        TEXT             NOT NULL,
    side          SMALLINT         NOT NULL,     -- -1 short | +1 long
    open_ts       TIMESTAMPTZ      NOT NULL,
    close_ts      TIMESTAMPTZ,
    entry_price   DOUBLE PRECISION,
    exit_price    DOUBLE PRECISION,
    size_lots     DOUBLE PRECISION,
    pnl           DOUBLE PRECISION,
    pnl_net       DOUBLE PRECISION,             -- after spread + commission + slippage
    meta_prob     DOUBLE PRECISION,             -- L5 P(correct) at entry
    regime        TEXT,                         -- L2 regime at entry
    spread_entry  DOUBLE PRECISION,
    slippage      DOUBLE PRECISION,
    status        TEXT             NOT NULL DEFAULT 'open',  -- open|closed|rejected
    meta          JSONB,
    created_at    TIMESTAMPTZ      NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS trades_symbol_open_idx
    ON market.trades (symbol, open_ts DESC);
