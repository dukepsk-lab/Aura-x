"""Layer 8 — alerts (Telegram + log).

Alerts for fills, risk breaches, regime flips and model decay feed the operator
and the VENUS-X dashboard. Delivery is behind a :class:`Notifier` protocol so the
:class:`~aurax.l8_monitoring.monitor.Monitor` is testable: tests inject a
recording notifier, live uses :class:`TelegramNotifier`. Message *formatting* is
pure and shared.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from ..logging import get_logger

log = get_logger(__name__)


class AlertKind(str, Enum):
    FILL = "fill"
    BREACH = "breach"
    REGIME_FLIP = "regime_flip"
    DECAY = "decay"


@dataclass
class TelegramConfig:
    bot_token: str = ""
    chat_id: str = ""

    @property
    def configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)


def format_alert(kind: AlertKind, message: str) -> str:
    """Render a compact, prefixed alert line."""
    icon = {
        AlertKind.FILL: "✅",
        AlertKind.BREACH: "🛑",
        AlertKind.REGIME_FLIP: "🔄",
        AlertKind.DECAY: "📉",
    }[kind]
    return f"{icon} [Aura-X][{kind.value}] {message}"


@runtime_checkable
class Notifier(Protocol):
    def send(self, kind: AlertKind, message: str) -> bool: ...


class LogNotifier:
    """Notifier that writes alerts to the structured log (default/dev)."""

    def send(self, kind: AlertKind, message: str) -> bool:
        log.info("alert", kind=kind.value, text=format_alert(kind, message))
        return True


class NullNotifier:
    """Silently drop alerts (disable alerting)."""

    def send(self, kind: AlertKind, message: str) -> bool:
        return False


class TelegramNotifier:
    """Deliver alerts to Telegram.

    ``transport`` (a ``callable(text) -> bool``) is injectable so delivery can be
    tested without a network; the default transport POSTs via ``httpx`` and is a
    graceful no-op when the bot token / chat id are not configured.
    """

    def __init__(
        self,
        config: TelegramConfig | None = None,
        *,
        transport: Callable[[str], bool] | None = None,
    ) -> None:
        self.config = config or TelegramConfig()
        self._transport = transport

    def send(self, kind: AlertKind, message: str) -> bool:
        return self._deliver(format_alert(kind, message))

    def _deliver(self, text: str) -> bool:
        if self._transport is not None:
            return bool(self._transport(text))
        if not self.config.configured:
            log.info("telegram_not_configured", text=text)
            return False
        try:  # pragma: no cover - network path
            import httpx

            url = f"https://api.telegram.org/bot{self.config.bot_token}/sendMessage"
            resp = httpx.post(url, json={"chat_id": self.config.chat_id, "text": text}, timeout=10.0)
            return resp.status_code == 200
        except Exception as exc:  # noqa: BLE001 - alerting must never raise into trading
            log.warning("telegram_send_failed", err=str(exc))
            return False
