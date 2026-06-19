"""Layer 8 — Monitoring & Lifecycle (Telegram · dashboard · decay detection)."""

from __future__ import annotations

from .alerts import AlertKind, TelegramConfig, TelegramNotifier, format_alert
from .decay import DecayConfig, DecayMonitor, calibration_error, rolling_sharpe

__all__ = [
    "TelegramNotifier",
    "TelegramConfig",
    "AlertKind",
    "format_alert",
    "DecayMonitor",
    "DecayConfig",
    "rolling_sharpe",
    "calibration_error",
]
