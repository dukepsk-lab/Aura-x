"""FastAPI surface for Aura-X (health · data · features · labels)."""

from __future__ import annotations

__all__ = ["create_app"]


def create_app():  # lazy so importing the package doesn't require FastAPI
    from .main import create_app as _factory

    return _factory()
