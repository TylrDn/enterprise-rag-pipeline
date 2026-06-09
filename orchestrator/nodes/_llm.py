"""Shared LLM construction for CRAG nodes.

Every node builds its chat model through :func:`get_chat_llm` so that Langfuse
callbacks (from :mod:`core.observability`) are attached consistently and the
model name comes from a single environment variable.
"""
from __future__ import annotations

import os

from langchain_openai import ChatOpenAI

from core.observability import get_callbacks

CHAT_MODEL = os.getenv("CHAT_MODEL", "meta/llama-3.1-70b-instruct")
NIM_BASE_URL = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")


def get_chat_llm(temperature: float = 0.0, max_tokens: int = 1024) -> ChatOpenAI:
    """Return a NIM-backed chat model with Langfuse tracing attached.

    Args:
        temperature: Sampling temperature.
        max_tokens: Maximum response tokens.

    Returns:
        ChatOpenAI: Configured chat model pointed at the NIM endpoint.
    """
    api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("NIM_API_KEY") or ""
    return ChatOpenAI(
        model=CHAT_MODEL,
        base_url=NIM_BASE_URL,
        api_key=api_key,  # type: ignore[arg-type]  # str coerced to SecretStr
        temperature=temperature,
        max_tokens=max_tokens,  # type: ignore[call-arg]
        callbacks=get_callbacks(),
    )
