"""Configuration loading.

Two distinct concerns, deliberately kept apart:

* **Secrets / connection details** (DB, MT5, Telegram, API) come from the
  environment via :class:`Settings` (pydantic-settings, ``AURAX_`` prefix,
  ``__`` nesting). Never hard-coded, never committed.
* **Algorithm parameters** (features, labeling, regime, risk thresholds) come
  from the versioned YAML in ``config/`` via :func:`load_params`. These belong
  in git because they define model behaviour and must be reproducible.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .enums import Environment
from .types import InstrumentSpec

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


class ConfigError(RuntimeError):
    """Raised when configuration is missing or malformed."""


# --- Secrets / connection settings (environment) -----------------------------
class DBSettings(BaseModel):
    host: str = "localhost"
    port: int = 5432
    name: str = "aurax"
    user: str = "aurax"
    password: str = "change-me"
    schema_: str = Field(default="market", alias="schema")

    @property
    def dsn(self) -> str:
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )

    @property
    def async_dsn(self) -> str:
        return self.dsn.replace("postgresql://", "postgresql+asyncpg://", 1)


class MT5Settings(BaseModel):
    login: int = 0
    password: str = ""
    server: str = ""
    terminal_path: str = ""


class TelegramSettings(BaseModel):
    bot_token: str = ""
    chat_id: str = ""


class APISettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class Settings(BaseSettings):
    """Environment-sourced settings. Loaded from process env and ``.env``."""

    model_config = SettingsConfigDict(
        env_prefix="AURAX_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: Environment = Environment.DEV
    log_level: str = "INFO"

    db: DBSettings = Field(default_factory=DBSettings)
    mt5: MT5Settings = Field(default_factory=MT5Settings)
    telegram: TelegramSettings = Field(default_factory=TelegramSettings)
    api: APISettings = Field(default_factory=APISettings)


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return process-wide cached :class:`Settings`."""
    return Settings()


# --- Algorithm parameters (YAML) ---------------------------------------------
def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


@functools.lru_cache(maxsize=8)
def load_params(config_dir: Path | str | None = None) -> dict[str, Any]:
    """Load and merge YAML config, resolving the ``includes:`` list.

    ``default.yaml`` lists sibling files under ``includes:``; each is merged in
    order, then the remaining top-level keys of ``default.yaml`` are overlaid.
    """
    cdir = Path(config_dir) if config_dir is not None else CONFIG_DIR
    root_file = cdir / "default.yaml"
    if not root_file.exists():
        raise ConfigError(f"missing config file: {root_file}")

    root = yaml.safe_load(root_file.read_text()) or {}
    merged: dict[str, Any] = {}
    for include in root.pop("includes", []) or []:
        inc_path = cdir / include
        if not inc_path.exists():
            raise ConfigError(f"included config not found: {inc_path}")
        merged = _deep_merge(merged, yaml.safe_load(inc_path.read_text()) or {})
    return _deep_merge(merged, root)


def get_instrument_specs(config_dir: Path | str | None = None) -> dict[str, InstrumentSpec]:
    """Build :class:`InstrumentSpec` objects from ``instruments.yaml``."""
    params = load_params(config_dir)
    specs: dict[str, InstrumentSpec] = {}
    for symbol, spec in params.get("instruments", {}).items():
        specs[symbol] = InstrumentSpec(
            symbol=spec["symbol"],
            pip_size=spec["pip_size"],
            digits=spec["digits"],
            contract_size=spec["contract_size"],
            min_lot=spec["min_lot"],
            lot_step=spec["lot_step"],
            base_ccy=spec["base_ccy"],
            quote_ccy=spec["quote_ccy"],
            asset_class=spec.get("asset_class", "fx_major"),
        )
    return specs
