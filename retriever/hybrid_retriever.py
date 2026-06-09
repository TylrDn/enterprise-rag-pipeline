"""Backward-compatible re-export of the canonical hybrid retriever.

The single implementation now lives in :mod:`pipeline.retriever`. This module
remains so existing imports (``from retriever.hybrid_retriever import
HybridRetriever``) keep working.
"""
from __future__ import annotations

from pipeline.retriever import HybridRetriever

__all__ = ["HybridRetriever"]
