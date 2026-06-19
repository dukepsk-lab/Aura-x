"""Layer 8 — Telegram alerts (scaffold).

Alerts for fills, risk breaches and regime flips feed the operator and the
VENUS-X dashboard. Sender is a thin stub here; wiring to python-telegram-bot is
the v1 build target. Message *formatting* is implemented (pure, testable).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AlertKind(str, Enum):
    FILL = "fill"
    BREACH = "breach"
    REGIME_FLIP = "regime_flip"
    DECAY = "decay"


@dataclass
class TelegramConfig:
    bot_token: str = ""
    chat_id: str = ""


def format_alert(kind: AlertKind, message: str) -> str:
    """Render a compact, prefixed alert line (implemented)."""
    icon = {
        AlertKind.FILL: "✅",
        AlertKind.BREACH: "🛑",
        AlertKind.REGIME_FLIP: "🔄",
        AlertKind.DECAY: "📉",
    }[kind]
    return f"{icon} [Aura-X][{kind.value}] {message}"


class TelegramNotifier:
    """Send alerts to Telegram."""

    def __init__(self, config: TelegramConfig | None = None) -> None:
        self.config = config or TelegramConfig()

    def send(self, kind: AlertKind, message: str) -> None:
        """Deliver an alert via the Telegram Bot API. TODO(v1)."""
        raise NotImplementedError("Telegram delivery lands in Roadmap v1 (L8).")
