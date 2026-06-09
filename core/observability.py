"""Observability utilities for enterprise-rag-pipeline.

Every LLM and embedding call in this repository routes its callbacks through
:func:`get_callbacks`. Tracing degrades gracefully: if the ``langfuse`` package
is missing or credentials are unset, the handler is ``None`` and callbacks are
an empty list, so no call site ever raises because of observability wiring.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional Langfuse import — tracing degrades gracefully if unavailable.
# ---------------------------------------------------------------------------
try:
    from langfuse.callback import CallbackHandler

    LANGFUSE_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without langfuse
    CallbackHandler = None  # type: ignore[assignment,misc]
    LANGFUSE_AVAILABLE = False


def get_langfuse_handler() -> "CallbackHandler | None":
    """Return a configured Langfuse callback handler, or ``None`` if disabled.

    Reads ``LANGFUSE_PUBLIC_KEY``, ``LANGFUSE_SECRET_KEY``, and ``LANGFUSE_HOST``
    from the environment. Logs a warning and returns ``None`` when the package
    is not installed or credentials are absent, so tracing never raises.

    Returns:
        CallbackHandler | None: A handler when configured, otherwise ``None``.
    """
    if not LANGFUSE_AVAILABLE:
        logger.warning("langfuse not installed — tracing disabled.")
        return None

    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")
    host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not public_key or not secret_key:
        logger.warning(
            "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set — tracing disabled."
        )
        return None

    return CallbackHandler(public_key=public_key, secret_key=secret_key, host=host)


def get_callbacks() -> list[Any]:
    """Return the list of active LangChain callbacks for an invocation.

    Returns:
        list[Any]: ``[langfuse_handler]`` when tracing is configured, else ``[]``.
    """
    handler = get_langfuse_handler()
    return [handler] if handler else []
