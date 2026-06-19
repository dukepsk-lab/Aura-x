"""Health & readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from ... import __version__
from ...config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok", "version": __version__, "env": get_settings().env.value}


@router.get("/ready")
def ready() -> dict[str, object]:
    """Readiness probe — reports whether optional subsystems are importable."""
    checks: dict[str, bool] = {}
    try:
        import sqlalchemy  # noqa: F401

        checks["db_driver"] = True
    except ModuleNotFoundError:
        checks["db_driver"] = False
    try:
        from ...l0_data import MT5Client

        checks["mt5"] = MT5Client().available
    except Exception:  # noqa: BLE001
        checks["mt5"] = False
    return {"ready": True, "checks": checks}
