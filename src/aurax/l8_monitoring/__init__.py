"""Layer 8 — Monitoring & Lifecycle (Telegram · dashboard · decay detection).

* :class:`Monitor` / :class:`MonitorConfig`  orchestration: alerts + decay + dashboard
* :class:`DashboardSnapshot` / :func:`journal_metrics`  the VENUS-X terminal feed
* alerts: :class:`TelegramNotifier`, :class:`LogNotifier`, :func:`format_alert`
* decay: :class:`DecayMonitor`, :class:`DecayReport`, :func:`rolling_sharpe`,
  :func:`calibration_error`
"""

from __future__ import annotations

from .alerts import (
    AlertKind,
    LogNotifier,
    Notifier,
    NullNotifier,
    TelegramConfig,
    TelegramNotifier,
    format_alert,
)
from .decay import (
    DecayConfig,
    DecayMonitor,
    DecayReport,
    calibration_error,
    rolling_sharpe,
)
from .monitor import DashboardSnapshot, Monitor, MonitorConfig, journal_metrics

__all__ = [
    "Monitor",
    "MonitorConfig",
    "DashboardSnapshot",
    "journal_metrics",
    "TelegramNotifier",
    "LogNotifier",
    "NullNotifier",
    "Notifier",
    "TelegramConfig",
    "AlertKind",
    "format_alert",
    "DecayMonitor",
    "DecayConfig",
    "DecayReport",
    "rolling_sharpe",
    "calibration_error",
]
