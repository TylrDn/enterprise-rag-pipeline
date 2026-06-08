"""Embedding model wrapper supporting NVIDIA NIM and HuggingFace backends.

Backend is selected via EMBEDDING_BACKEND env var:
  - 'nim'         → NVIDIA NIM embeddings endpoint (OpenAI-compatible)
  - 'huggingface' → local HuggingFace sentence-transformers model

All calls are traced via Langfuse for observability.
"""

from __future__ import annotations

import os
import time
from typing import List

from dotenv import load_dotenv
from langfuse import Langfuse
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# ---------------------------------------------------------------------------
# Langfuse client (no-op if keys not set)
# ---------------------------------------------------------------------------
_langfuse: Langfuse | None = None


def _get_langfuse() -> Langfuse | None:
    if os.getenv("LANGFUSE_SECRET_KEY") and os.getenv("LANGFUSE_PUBLIC_KEY"):
        global _langfuse
        if _langfuse is None:
            _langfuse = Langfuse(
                secret_key=os.environ["LANGFUSE_SECRET_KEY"],
                public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
                host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            )
        return _langfuse
    return None


# ---------------------------------------------------------------------------
# NIM Embeddings wrapper
# ---------------------------------------------------------------------------

class NIMEmbeddings(Embeddings):
    """Thin wrapper around NVIDIA NIM embeddings via OpenAI-compatible API."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model = model or os.getenv(
            "NIM_EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5"
        )
        self.base_url = base_url or os.getenv(
            "NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"
        )
        self.api_key = api_key or os.environ["NIM_API_KEY"]
        self._client = OpenAIEmbeddings(
            model=self.model,
            openai_api_base=self.base_url,
            openai_api_key=self.api_key,
            check_embedding_ctx_length=False,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._client.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._client.embed_query(text)


# ---------------------------------------------------------------------------
# HuggingFace local embeddings wrapper
# ---------------------------------------------------------------------------

class HuggingFaceEmbeddings(Embeddings):
    """Local sentence-transformers embedding model."""

    def __init__(self, model_name: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for HuggingFace backend. "
                "Run: pip install sentence-transformers"
            ) from exc

        self.model_name = model_name or os.getenv(
            "HF_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5"
        )
        self._model = SentenceTransformer(self.model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._model.encode([text], normalize_embeddings=True)[0].tolist()


# ---------------------------------------------------------------------------
# Traced embedder — wraps any Embeddings backend with Langfuse spans
# ---------------------------------------------------------------------------

class TracedEmbedder(Embeddings):
    """Wraps an Embeddings backend and emits Langfuse spans for every call."""

    def __init__(self, backend: Embeddings, backend_name: str) -> None:
        self._backend = backend
        self._backend_name = backend_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        lf = _get_langfuse()
        start = time.perf_counter()
        try:
            vectors = self._backend.embed_documents(texts)
        finally:
            elapsed = time.perf_counter() - start
            if lf:
                lf.generation(
                    name="embed_documents",
                    model=self._backend_name,
                    input={"num_texts": len(texts)},
                    output={"num_vectors": len(vectors) if "vectors" in dir() else 0},
                    metadata={"latency_s": round(elapsed, 4)},
                )
        return vectors

    def embed_query(self, text: str) -> List[float]:
        lf = _get_langfuse()
        start = time.perf_counter()
        try:
            vector = self._backend.embed_query(text)
        finally:
            elapsed = time.perf_counter() - start
            if lf:
                lf.generation(
                    name="embed_query",
                    model=self._backend_name,
                    input={"text_length": len(text)},
                    output={"vector_dim": len(vector) if "vector" in dir() else 0},
                    metadata={"latency_s": round(elapsed, 4)},
                )
        return vector


# ---------------------------------------------------------------------------
# Factory — the primary public interface
# ---------------------------------------------------------------------------

def get_embedder(backend: str | None = None) -> Embeddings:
    """Return a traced Embeddings instance based on EMBEDDING_BACKEND env var.

    Args:
        backend: Override the EMBEDDING_BACKEND env var. Options: 'nim', 'huggingface'.

    Returns:
        A TracedEmbedder wrapping the selected backend.

    Raises:
        ValueError: If an unknown backend is specified.
    """
    resolved = (backend or os.getenv("EMBEDDING_BACKEND", "nim")).lower()

    if resolved == "nim":
        inner: Embeddings = NIMEmbeddings()
        name = f"nim/{os.getenv('NIM_EMBEDDING_MODEL', 'nvidia/nv-embedqa-e5-v5')}"
    elif resolved == "huggingface":
        inner = HuggingFaceEmbeddings()
        name = f"hf/{os.getenv('HF_EMBEDDING_MODEL', 'BAAI/bge-base-en-v1.5')}"
    else:
        raise ValueError(
            f"Unknown embedding backend: '{resolved}'. Choose 'nim' or 'huggingface'."
        )

    return TracedEmbedder(backend=inner, backend_name=name)
