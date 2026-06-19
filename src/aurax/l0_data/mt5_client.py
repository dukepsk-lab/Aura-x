"""Thin wrapper over the MetaTrader5 terminal API.

The ``MetaTrader5`` package ships a Windows-only wheel. To keep the rest of
Layer 0 (and all of L1/L4) developable and testable on any OS, the import is
*soft*: the module always imports, and the hard failure is deferred to
:meth:`MT5Client.connect`, raising a clear :class:`MT5NotAvailableError`.

The raw rate/tick payloads returned here are deliberately left in MT5's native
shape; normalisation into the canonical schema lives in :mod:`aurax.l0_data.schema`
so it can be unit-tested without a terminal.
"""

from __future__ import annotations

from datetime import datetime
from types import TracebackType
from typing import Any

from ..config import Settings, get_settings
from ..enums import Timeframe
from ..logging import get_logger

log = get_logger(__name__)

try:  # pragma: no cover - platform dependent
    import MetaTrader5 as _mt5  # type: ignore

    _MT5_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001 - any import failure means "unavailable"
    _mt5 = None  # type: ignore
    _MT5_IMPORT_ERROR = exc


# Map our Timeframe enum → MetaTrader5 timeframe constants (resolved lazily).
_TIMEFRAME_CONST: dict[Timeframe, str] = {
    Timeframe.M15: "TIMEFRAME_M15",
    Timeframe.H4: "TIMEFRAME_H4",
    Timeframe.D1: "TIMEFRAME_D1",
}


class MT5NotAvailableError(RuntimeError):
    """Raised when the MetaTrader5 terminal/package is not usable here."""


class MT5Client:
    """Connect to an MT5 terminal and pull rates/ticks.

    Usable as a context manager::

        with MT5Client() as client:
            rates = client.copy_rates("EURUSD", Timeframe.H4, start, end)
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._connected = False

    # --- availability --------------------------------------------------------
    @property
    def available(self) -> bool:
        """Whether the MetaTrader5 package imported successfully."""
        return _mt5 is not None

    def _require(self) -> Any:
        if _mt5 is None:
            raise MT5NotAvailableError(
                "MetaTrader5 package is not importable on this platform. "
                "Install on a Windows host with:  pip install -e '.[mt5]'. "
                f"(original import error: {_MT5_IMPORT_ERROR})"
            )
        return _mt5

    # --- lifecycle -----------------------------------------------------------
    def connect(self) -> None:
        """Initialise the terminal and log in."""
        mt5 = self._require()
        cfg = self._settings.mt5
        kwargs: dict[str, Any] = {}
        if cfg.terminal_path:
            kwargs["path"] = cfg.terminal_path
        if cfg.login:
            kwargs.update(login=cfg.login, password=cfg.password, server=cfg.server)

        if not mt5.initialize(**kwargs):
            raise MT5NotAvailableError(f"mt5.initialize failed: {mt5.last_error()}")
        self._connected = True
        log.info("mt5_connected", server=cfg.server or "default", login=cfg.login or None)

    def shutdown(self) -> None:
        if self._connected and _mt5 is not None:
            _mt5.shutdown()
            self._connected = False
            log.info("mt5_shutdown")

    def __enter__(self) -> MT5Client:
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.shutdown()

    # --- data pulls ----------------------------------------------------------
    def _timeframe(self, timeframe: Timeframe) -> int:
        mt5 = self._require()
        return getattr(mt5, _TIMEFRAME_CONST[timeframe])

    def copy_rates(
        self, symbol: str, timeframe: Timeframe, start: datetime, end: datetime
    ) -> Any:
        """Return raw MT5 rates for ``[start, end]`` (native structured array)."""
        mt5 = self._require()
        if not self._connected:
            self.connect()
        rates = mt5.copy_rates_range(symbol, self._timeframe(timeframe), start, end)
        if rates is None:
            raise MT5NotAvailableError(
                f"copy_rates_range returned None for {symbol} {timeframe}: {mt5.last_error()}"
            )
        return rates

    def copy_ticks(self, symbol: str, start: datetime, end: datetime) -> Any:
        """Return raw MT5 ticks for ``[start, end]`` (native structured array)."""
        mt5 = self._require()
        if not self._connected:
            self.connect()
        # COPY_TICKS_ALL captures both bid/ask changes — needed for spread history.
        ticks = mt5.copy_ticks_range(symbol, start, end, mt5.COPY_TICKS_ALL)
        if ticks is None:
            raise MT5NotAvailableError(
                f"copy_ticks_range returned None for {symbol}: {mt5.last_error()}"
            )
        return ticks
