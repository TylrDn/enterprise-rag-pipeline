"""Chunk, deduplicate, and upsert documents into a vector store backend."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.exceptions import IngestionError

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))
DEFAULT_BATCH_SIZE = 64


@dataclass
class IndexResult:
    """Outcome of an indexing run."""

    total: int = 0
    indexed: int = 0
    skipped: int = 0
    failed: int = 0
    dry_run: bool = False
    ids: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """True when no chunk batches failed to upsert."""
        return self.failed == 0

    def __str__(self) -> str:
        return (
            f"IndexResult(total={self.total}, indexed={self.indexed}, "
            f"skipped={self.skipped}, failed={self.failed}, dry_run={self.dry_run})"
        )


def _content_hash(text: str) -> str:
    """Return a stable 16-character hash of ``text``."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _chunk_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Split documents into chunks, stamping ``chunk_index`` and ``chunk_hash``.

    The hash incorporates the chunk index so positionally distinct chunks with
    identical text remain individually addressable.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
        chunk.metadata["chunk_hash"] = _content_hash(f"{index}:{chunk.page_content}")
    return chunks


def _deduplicate(
    chunks: list[Document], existing_hashes: set[str] | None = None
) -> tuple[list[Document], set[str]]:
    """Drop chunks whose hash is already known.

    Args:
        chunks: Candidate chunks (must carry ``chunk_hash`` metadata).
        existing_hashes: Hashes already present in the store, if any.

    Returns:
        tuple: ``(new_chunks, seen_hashes)``.
    """
    seen: set[str] = set(existing_hashes) if existing_hashes else set()
    new_chunks: list[Document] = []
    for chunk in chunks:
        digest = chunk.metadata.get("chunk_hash") or _content_hash(chunk.page_content)
        if digest in seen:
            continue
        seen.add(digest)
        new_chunks.append(chunk)
    return new_chunks, seen


def _load_config(config_path: str | Path | None) -> dict[str, Any]:
    """Load a pipeline YAML config, returning an empty dict when absent."""
    if config_path is None:
        return {}
    path = Path(config_path)
    if not path.exists():
        logger.warning("Config %s not found; using defaults.", path)
        return {}
    with open(path) as handle:
        return yaml.safe_load(handle) or {}


def _batched(items: list[Document], size: int) -> list[list[Document]]:
    """Split ``items`` into batches of at most ``size``."""
    return [items[i : i + size] for i in range(0, len(items), size)]


class Indexer:
    """Chunk, deduplicate, and upsert documents into a vector store backend."""

    def __init__(
        self,
        backend: Any,
        config_path: str | Path | None = None,
        embedder: Any | None = None,
    ) -> None:
        """Initialize the indexer.

        Args:
            backend: A vector store backend exposing ``upsert_documents``.
            config_path: Optional path to a pipeline YAML config.
            embedder: Optional embeddings instance (unused by most backends).
        """
        self.backend = backend
        self.embedder = embedder
        config = _load_config(config_path)
        chunking = config.get("chunking", {})
        embedding = config.get("embedding", {})
        self.chunk_size = int(chunking.get("chunk_size", DEFAULT_CHUNK_SIZE))
        self.chunk_overlap = int(chunking.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP))
        self.batch_size = int(embedding.get("batch_size", DEFAULT_BATCH_SIZE))

    def index(
        self,
        documents: list[Document],
        dry_run: bool = False,
        existing_hashes: set[str] | None = None,
    ) -> IndexResult:
        """Chunk, deduplicate, and upsert ``documents``.

        Args:
            documents: Raw documents to index.
            dry_run: When True, chunk and dedup but skip the upsert.
            existing_hashes: Known chunk hashes to skip (idempotent re-ingest).

        Returns:
            IndexResult: Counts of total/indexed/skipped/failed chunks.
        """
        if not documents:
            return IndexResult(total=0, indexed=0, skipped=0, failed=0, dry_run=dry_run)

        chunks = _chunk_documents(documents, self.chunk_size, self.chunk_overlap)
        new_chunks, _ = _deduplicate(chunks, existing_hashes)
        total = len(chunks)
        skipped = total - len(new_chunks)

        if dry_run:
            logger.info("Dry run: %d chunks, %d new, %d skipped", total, len(new_chunks), skipped)
            return IndexResult(total=total, indexed=0, skipped=skipped, dry_run=True)

        if not new_chunks:
            return IndexResult(total=total, indexed=0, skipped=skipped)

        indexed = 0
        failed = 0
        ids: list[str] = []
        for batch in _batched(new_chunks, self.batch_size):
            try:
                batch_ids = self.backend.upsert_documents(batch)
                ids.extend(batch_ids or [])
                indexed += len(batch)
            except Exception:
                logger.exception("Failed to upsert batch of %d chunks", len(batch))
                failed += len(batch)

        return IndexResult(
            total=total, indexed=indexed, skipped=skipped, failed=failed, ids=ids
        )

    def index_from_loader(self, loader_callable: Any, *args: Any, **kwargs: Any) -> IndexResult:
        """Run a (possibly failing) loader, then index its documents.

        Args:
            loader_callable: A callable returning ``list[Document]``.

        Raises:
            IngestionError: If the loader raises while reading its source.
        """
        try:
            documents = loader_callable(*args, **kwargs)
        except Exception as exc:
            name = getattr(loader_callable, "__name__", loader_callable)
            logger.exception("Loader %s failed", name)
            raise IngestionError(str(exc)) from exc
        return self.index(documents)

    async def index_async(
        self,
        documents: list[Document],
        dry_run: bool = False,
        existing_hashes: set[str] | None = None,
    ) -> IndexResult:
        """Non-blocking wrapper around :meth:`index`."""
        return await asyncio.to_thread(self.index, documents, dry_run, existing_hashes)

    async def index_from_loader_async(
        self, loader_callable: Any, *args: Any, **kwargs: Any
    ) -> IndexResult:
        """Run a sync loader in a thread pool, then index its documents."""
        try:
            documents = await asyncio.to_thread(loader_callable, *args, **kwargs)
        except Exception as exc:
            name = getattr(loader_callable, "__name__", loader_callable)
            logger.exception("Loader %s failed", name)
            raise IngestionError(str(exc)) from exc
        return await self.index_async(documents)
