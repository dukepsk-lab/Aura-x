"""Aura-X — ML-driven adaptive trading system (EURUSD/GBPUSD · H4 · MT5).

The package is organised as the 8-layer pipeline described in
``docs/architecture.md``:

    l0_data      Layer 0  Data ingestion & TimescaleDB storage
    l1_features  Layer 1  Feature engineering (volatility/trend/cross-pair/session)
    l2_regime    Layer 2  Regime detection (HMM + Hurst/KER gating)
    l3_primary   Layer 3  Primary SIDE model (CNN + GBM ensemble)
    l4_labeling  Layer 4  Triple-barrier labels + sample-uniqueness weights
    l5_meta      Layer 5  Meta-label TRUST model (calibrated stacking)
    l6_risk      Layer 6  Risk & position sizing (+ L6b allocation)
    l7_execution Layer 7  MT5 execution with pre-trade guards
    l8_monitoring Layer 8 Monitoring, alerts, model-decay detection
    validation   CPCV / deflated-Sharpe / walk-forward gate
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
