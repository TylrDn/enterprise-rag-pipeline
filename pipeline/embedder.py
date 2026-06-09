"""Embedding model wrapper — NIM OpenAI-compatible endpoint with retries."""
from __future__ import annotations

import logging
import os
from typing import List

from langchain_openai import OpenAIEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential

from core.exceptions import EmbeddingError

logger = logging.getLogger(__name__)

NIM_BASE_URL = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
NIM_API_KEY = os.getenv("NVIDIA_API_KEY") or os.getenv("NIM_API_KEY") or ""
EMBED_MODEL: str = (
    os.getenv("NIM_EMBEDDING_MODEL") or os.getenv("EMBED_MODEL") or "nvidia/nv-embedqa-e5-v5"
)

_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)


class Embedder:
    """Thin wrapper around the NIM embedding endpoint."""

    def __init__(self, model: str = EMBED_MODEL) -> None:
        """Initialize the embedder.

        Args:
            model: NIM embedding model name.
        """
        self.model = model
        self._client = OpenAIEmbeddings(
            model=model,
            base_url=NIM_BASE_URL,
            api_key=NIM_API_KEY,  # type: ignore[arg-type]  # accepts str; coerced to SecretStr
        )

    @_RETRY
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of documents, retrying transient failures.

        Raises:
            EmbeddingError: If embedding fails after all retries.
        """
        try:
            return self._client.embed_documents(texts)
        except Exception as exc:
            logger.exception("Embedding documents failed")
            raise EmbeddingError(str(exc)) from exc

    @_RETRY
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query, retrying transient failures.

        Raises:
            EmbeddingError: If embedding fails after all retries.
        """
        try:
            return self._client.embed_query(text)
        except Exception as exc:
            logger.exception("Embedding query failed")
            raise EmbeddingError(str(exc)) from exc

    def as_langchain(self) -> OpenAIEmbeddings:
        """Return the underlying LangChain embeddings object."""
        return self._client


def get_embedder(model: str | None = None) -> Embedder:
    """Return a configured :class:`Embedder`.

    Args:
        model: Optional model override; defaults to ``NIM_EMBEDDING_MODEL``.
    """
    return Embedder(model=model or EMBED_MODEL)
