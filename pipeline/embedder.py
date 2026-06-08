"""Embedding model wrapper — NIM or HuggingFace, drop-in swappable."""
from __future__ import annotations

import os
from typing import List

from langchain_openai import OpenAIEmbeddings

NIM_BASE_URL = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
NIM_API_KEY = os.getenv("NVIDIA_API_KEY", "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")


class Embedder:
    """Thin wrapper around NIM embedding endpoint."""

    def __init__(self, model: str = EMBED_MODEL) -> None:
        self.model = model
        self._client = OpenAIEmbeddings(
            model=model,
            openai_api_base=NIM_BASE_URL,
            openai_api_key=NIM_API_KEY,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._client.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._client.embed_query(text)

    def as_langchain(self) -> OpenAIEmbeddings:
        return self._client
