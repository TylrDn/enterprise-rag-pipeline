"""NVIDIA NIM embedding client with batching."""
from openai import OpenAI
from langchain_core.embeddings import Embeddings
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()


class NIMEmbedder(Embeddings):
    """LangChain-compatible NIM embedding client."""

    def __init__(self, model: str = None, batch_size: int = 32):
        self.model = model or os.getenv("EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5")
        self.batch_size = batch_size
        self.client = OpenAI(
            api_key=os.getenv("NVIDIA_API_KEY"),
            base_url=os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        )

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            input=texts,
            model=self.model,
            encoding_format="float",
            extra_body={"input_type": "query", "truncate": "END"},
        )
        return [d.embedding for d in response.data]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            results.extend(self._embed_batch(batch))
        return results

    def embed_query(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            input=[text],
            model=self.model,
            encoding_format="float",
            extra_body={"input_type": "query", "truncate": "END"},
        )
        return response.data[0].embedding
