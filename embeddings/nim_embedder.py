"""NVIDIA NIM embedding client with batching and retries."""
from __future__ import annotations

import logging
import os
from typing import List

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from core.exceptions import EmbeddingError

load_dotenv()
logger = logging.getLogger(__name__)

_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)


class NIMEmbedder(Embeddings):
    """LangChain-compatible NIM embedding client."""

    def __init__(self, model: str | None = None, batch_size: int = 32) -> None:
        """Initialize the embedder.

        Args:
            model: NIM embedding model name. Defaults to ``NIM_EMBEDDING_MODEL``.
            batch_size: Number of texts per embedding request.
        """
        self.model: str = (
            model or os.getenv("NIM_EMBEDDING_MODEL") or "nvidia/nv-embedqa-e5-v5"
        )
        self.batch_size = batch_size
        self.client = OpenAI(
            api_key=os.getenv("NVIDIA_API_KEY") or os.getenv("NIM_API_KEY"),
            base_url=os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        )

    @_RETRY
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model,
                encoding_format="float",
                extra_body={"input_type": "query", "truncate": "END"},
            )
        except Exception as exc:
            logger.exception("NIM embedding request failed")
            raise EmbeddingError(str(exc)) from exc
        return [d.embedding for d in response.data]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            results.extend(self._embed_batch(batch))
        return results

    @_RETRY
    def embed_query(self, text: str) -> List[float]:
        try:
            response = self.client.embeddings.create(
                input=[text],
                model=self.model,
                encoding_format="float",
                extra_body={"input_type": "query", "truncate": "END"},
            )
        except Exception as exc:
            logger.exception("NIM embedding request failed")
            raise EmbeddingError(str(exc)) from exc
        return response.data[0].embedding
