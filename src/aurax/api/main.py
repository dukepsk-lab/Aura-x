"""FastAPI application factory for Aura-X.

Exposes read surfaces over the implemented layers (data / features / labels)
plus health. Requires the ``api`` extra (``pip install -e '.[api]'``); DB-backed
routes additionally need ``[db]``.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from ..config import get_settings
from ..logging import configure_logging, get_logger
from .routes import data, features, health, labels, monitor

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    log.info("aurax_api_start", env=settings.env.value)
    yield
    log.info("aurax_api_stop")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Aura-X API",
        version="0.1.0",
        summary="ML-driven adaptive trading system (EURUSD/GBPUSD · H4 · MT5)",
        lifespan=lifespan,
    )
    app.include_router(health.router)
    app.include_router(data.router)
    app.include_router(features.router)
    app.include_router(labels.router)
    app.include_router(monitor.router)
    return app


app = create_app()
