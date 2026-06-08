"""Embedding model wrapper — NIM endpoint or HuggingFace fallback."""
from langchain.embeddings.base import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
import os


class NIMEmbedder(Embeddings):
    """Routes to NVIDIA NIM embedding endpoint (OpenAI-compatible)."""

    def __init__(self, model: str = "nvidia/nv-embedqa-e5-v5", base_url: str | None = None):
        self._client = OpenAIEmbeddings(
            model=model,
            openai_api_key=os.environ["NVIDIA_API_KEY"],
            openai_api_base=base_url or os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._client.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._client.embed_query(text)


def get_embedder(backend: str = "nim", model: str | None = None) -> Embeddings:
    """Factory: returns NIM or HuggingFace embedder based on config."""
    if backend == "nim":
        return NIMEmbedder(model=model or "nvidia/nv-embedqa-e5-v5")
    elif backend == "huggingface":
        return HuggingFaceEmbeddings(model_name=model or "BAAI/bge-base-en-v1.5")
    raise ValueError(f"Unknown embedding backend: {backend}")
