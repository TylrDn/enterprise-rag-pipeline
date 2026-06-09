"""Deprecated entry point.

The canonical FastAPI application now lives in :mod:`api.server`. This module
re-exports it so ``api.main:app`` keeps working for any external references.
Use ``uvicorn api.server:app`` going forward.
"""
from __future__ import annotations

from api.server import app

__all__ = ["app"]
