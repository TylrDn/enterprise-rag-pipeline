"""NIM embedding client and document chunking utilities."""
from embeddings.chunker import markdown_chunk, recursive_chunk
from embeddings.nim_embedder import NIMEmbedder

__all__ = ["NIMEmbedder", "recursive_chunk", "markdown_chunk"]
