"""Custom exception hierarchy for the RAG pipeline.

These let callers distinguish failure domains (ingestion vs. embedding vs.
retrieval vs. generation) instead of catching bare ``Exception``.
"""
from __future__ import annotations


class RAGPipelineError(Exception):
    """Base class for all pipeline-specific errors."""


class IngestionError(RAGPipelineError):
    """Raised when a document loader fails to read or parse a source."""


class EmbeddingError(RAGPipelineError):
    """Raised when the NIM embedding endpoint fails after retries."""


class RetrievalError(RAGPipelineError):
    """Raised when hybrid retrieval cannot return candidate documents."""


class GenerationError(RAGPipelineError):
    """Raised when answer generation fails or returns an unusable result."""
