"""L8 — alerts, decay detection, and the Monitor orchestrator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aurax.enums import Regime
from aurax.l8_monitoring import (
    AlertKind,
    DecayConfig,
    DecayMonitor,
    LogNotifier,
    Monitor,
    MonitorConfig,
    TelegramConfig,
    TelegramNotifier,
    calibration_error,
    format_alert,
    journal_metrics,
)


class RecordingNotifier:
    """Test notifier that captures every alert."""

    def __init__(self):
        self.sent: list[tuple[AlertKind, str]] = []

    def send(self, kind: AlertKind, message: str) -> bool:
        self.sent.append((kind, message))
        return True


# --- alerts ------------------------------------------------------------------
def test_format_alert_prefixes_and_icons():
    assert format_alert(AlertKind.FILL, "x").startswith("✅ [Aura-X][fill]")
    assert format_alert(AlertKind.BREACH, "x").startswith("🛑 [Aura-X][breach]")


def test_telegram_notifier_transport_and_noop():
    captured = []
    notifier = TelegramNotifier(transport=lambda text: captured.append(text) or True)
    assert notifier.send(AlertKind.DECAY, "drift") is True
    assert captured[0] == format_alert(AlertKind.DECAY, "drift")
    # unconfigured (no token) and no transport → graceful no-op
    assert TelegramNotifier(TelegramConfig()).send(AlertKind.FILL, "x") is False


def test_log_notifier_returns_true():
    assert LogNotifier().send(AlertKind.FILL, "filled") is True


# --- decay detection ---------------------------------------------------------
def test_decay_report_flags_sharpe_and_calibration():
    mon = DecayMonitor(DecayConfig(sharpe_window=20, min_rolling_sharpe=0.0, max_calibration_error=0.10))
    rng = np.random.default_rng(0)
    well_prob = np.full(200, 0.6)
    well_out = (rng.uniform(size=200) < 0.6).astype(int)   # calibrated to 0.6

    # negative-edge returns (with variance) → negative rolling Sharpe → retrain
    losing = pd.Series(rng.normal(-0.002, 0.001, 40))
    rep = mon.evaluate(losing, well_prob, well_out)
    assert rep.sharpe_breached and rep.retrain_due and rep.reasons()

    # positive-edge + calibrated → no retrain
    winning = pd.Series(rng.normal(0.002, 0.001, 40))
    assert not mon.evaluate(winning, well_prob, well_out).retrain_due

    # miscalibrated meta-probs (0.6 predicted, never wins) → calibration breach
    miscal = mon.evaluate(winning, well_prob, np.zeros(200))
    assert miscal.calibration_breached


def test_calibration_error_zero_when_perfect():
    prob = np.array([0.0, 0.0, 1.0, 1.0])
    outcome = np.array([0, 0, 1, 1])
    assert calibration_error(prob, outcome) == 0.0


# --- journal metrics ---------------------------------------------------------
def _trades(n=80, seed=0):
    rng = np.random.default_rng(seed)
    pnl = rng.normal(5.0, 50.0, n)                  # slight positive edge
    meta = np.clip(rng.normal(0.6, 0.1, n), 0, 1)
    return pd.DataFrame({"pnl_net": pnl, "meta_prob": meta, "side": rng.choice([-1, 1], n)})


def test_journal_metrics():
    m = journal_metrics(_trades(), window=30)
    assert m["n_trades"] == 80
    assert 0.0 <= m["hit_rate"] <= 1.0
    assert np.isfinite(m["calibration_error"])
    # empty / missing tolerated
    assert journal_metrics(None)["n_trades"] == 0
    assert journal_metrics(pd.DataFrame())["n_trades"] == 0


# --- Monitor orchestrator ----------------------------------------------------
def test_monitor_regime_flip_and_drawdown_alerts():
    rec = RecordingNotifier()
    mon = Monitor(MonitorConfig(drawdown_alert=0.10), notifier=rec)

    assert mon.on_regime(Regime.TREND) is False     # first regime → baseline, no flip
    assert mon.on_regime(Regime.TREND) is False     # unchanged
    assert mon.on_regime(Regime.SHOCK) is True      # flip → alert
    assert any(k is AlertKind.REGIME_FLIP for k, _ in rec.sent)

    assert mon.update_equity(100_000) is False       # sets peak
    assert mon.update_equity(95_000) is False        # 5% DD
    assert mon.update_equity(88_000) is True         # 12% DD → breach alert
    assert any(k is AlertKind.BREACH for k, _ in rec.sent)


def test_monitor_on_fill_and_check_decay_alert():
    rec = RecordingNotifier()
    mon = Monitor(MonitorConfig(decay=DecayConfig(sharpe_window=20)), notifier=rec)
    mon.on_fill(symbol="EURUSD", side=1, lots=1.0, price=1.10000)
    assert rec.sent[-1][0] is AlertKind.FILL

    report = mon.check_decay(pd.Series(np.full(30, -0.001)), np.array([0.6] * 4), np.array([0, 0, 0, 0]))
    assert report.retrain_due
    assert any(k is AlertKind.DECAY for k, _ in rec.sent)


def test_monitor_dashboard_snapshot():
    mon = Monitor(MonitorConfig())
    mon.update_equity(100_000)
    snap = mon.snapshot(equity=92_000, trades=_trades(), regime=Regime.RANGE,
                        open_positions={"EURUSD": 1.0})
    assert snap.drawdown == pytest.approx(0.08, abs=1e-9)
    assert snap.n_trades == 80
    assert snap.regime == "range"
    assert "equity" in snap.as_dict() and snap.as_dict()["open_positions"] == {"EURUSD": 1.0}
