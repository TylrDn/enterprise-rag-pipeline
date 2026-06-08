"""Document indexer for the enterprise RAG pipeline.

Responsibilities:
  1. Split raw documents into chunks using configurable text splitters.
  2. Deduplicate chunks by content hash to avoid re-indexing unchanged docs.
  3. Batch-upsert chunks into the configured vector backend (pgvector or Milvus).
  4. Expose a simple index() entrypoint that orchestrates the full flow.

Configuration is driven by configs/pipeline.yaml (chunk_size, chunk_overlap,
batch_size) and the VECTOR_BACKEND env var ('pgvector' | 'milvus').
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import List, Optional

import yaml
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "pipeline.yaml"


def _load_config() -> dict:
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _content_hash(text: str) -> str:
    """Return a short SHA-256 hex digest for deduplication."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _chunk_documents(
    documents: List[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    """Split documents into fixed-size overlapping chunks.

    Preserves all original metadata and appends chunk-level fields:
      - chunk_index: position of this chunk within its source document
      - chunk_hash:  SHA-256 prefix for deduplication
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    chunks: List[Document] = []
    for doc_idx, doc in enumerate(documents):
        splits = splitter.split_documents([doc])
        for chunk_idx, chunk in enumerate(splits):
            chunk.metadata["chunk_index"] = chunk_idx
            chunk.metadata["chunk_hash"] = _content_hash(chunk.page_content)
            chunk.metadata.setdefault("doc_index", doc_idx)
        chunks.extend(splits)
    return chunks


def _deduplicate(
    chunks: List[Document],
    existing_hashes: Optional[set] = None,
) -> tuple[List[Document], set]:
    """Remove chunks whose content hash already exists in the store.

    Args:
        chunks:          Candidate chunks to index.
        existing_hashes: Set of hashes already present in the vector store.
                         Pass None to skip deduplication (index everything).

    Returns:
        Tuple of (new_chunks, seen_hashes) where seen_hashes is the updated set.
    """
    if existing_hashes is None:
        return chunks, {c.metadata["chunk_hash"] for c in chunks}

    new_chunks: List[Document] = []
    seen = set(existing_hashes)
    for chunk in chunks:
        h = chunk.metadata["chunk_hash"]
        if h not in seen:
            new_chunks.append(chunk)
            seen.add(h)
    skipped = len(chunks) - len(new_chunks)
    if skipped:
        logger.info("Deduplication: skipped %d already-indexed chunks.", skipped)
    return new_chunks, seen


def _batch(items: list, size: int):
    """Yield successive batches of `size` from `items`."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------

class Indexer:
    """Orchestrates chunking, deduplication, and upsert for a vector backend.

    Args:
        backend:      A PgVectorBackend or MilvusBackend instance.
        config_path:  Override path to pipeline.yaml.
    """

    def __init__(
        self,
        backend,  # PgVectorBackend | MilvusBackend
        config_path: Optional[Path] = None,
    ) -> None:
        self._backend = backend
        cfg = _load_config() if config_path is None else self._read_config(config_path)
        chunking = cfg.get("chunking", {})
        embedding = cfg.get("embedding", {})

        self.chunk_size: int = int(chunking.get("chunk_size", 1000))
        self.chunk_overlap: int = int(chunking.get("chunk_overlap", 200))
        self.batch_size: int = int(embedding.get("batch_size", 64))

    @staticmethod
    def _read_config(path: Path) -> dict:
        with open(path) as f:
            return yaml.safe_load(f) or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index(
        self,
        documents: List[Document],
        existing_hashes: Optional[set] = None,
        dry_run: bool = False,
    ) -> IndexResult:
        """Chunk, deduplicate, and upsert documents into the vector backend.

        Args:
            documents:       Raw documents to index (pre-loaded, not yet chunked).
            existing_hashes: Optional set of chunk_hash values already in the
                             store.  Pass None to skip dedup and index everything.
            dry_run:         If True, perform chunking and dedup but skip the
                             actual upsert.  Useful for previewing index changes.

        Returns:
            IndexResult with counts of total, new, skipped, and failed chunks.
        """
        if not documents:
            logger.warning("index() called with empty document list.")
            return IndexResult(total=0, indexed=0, skipped=0, failed=0)

        # Step 1 — chunk
        chunks = _chunk_documents(documents, self.chunk_size, self.chunk_overlap)
        logger.info("Chunked %d docs into %d chunks.", len(documents), len(chunks))

        # Step 2 — deduplicate
        new_chunks, _ = _deduplicate(chunks, existing_hashes)
        skipped = len(chunks) - len(new_chunks)

        if dry_run:
            logger.info("Dry run: would index %d chunks (%d skipped).", len(new_chunks), skipped)
            return IndexResult(
                total=len(chunks),
                indexed=0,
                skipped=skipped,
                failed=0,
                dry_run=True,
            )

        # Step 3 — batch upsert
        indexed = 0
        failed = 0
        for batch in _batch(new_chunks, self.batch_size):
            try:
                self._backend.upsert_documents(batch)
                indexed += len(batch)
                logger.debug("Upserted batch of %d chunks.", len(batch))
            except Exception as exc:  # noqa: BLE001
                logger.error("Batch upsert failed: %s", exc)
                failed += len(batch)

        logger.info(
            "Index complete: %d indexed, %d skipped, %d failed (of %d total chunks).",
            indexed, skipped, failed, len(chunks),
        )
        return IndexResult(
            total=len(chunks),
            indexed=indexed,
            skipped=skipped,
            failed=failed,
        )


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field  # noqa: E402


@dataclass
class IndexResult:
    """Summary of an indexing run."""

    total: int
    indexed: int
    skipped: int
    failed: int
    dry_run: bool = False

    @property
    def success(self) -> bool:
        """True if no chunks failed to upsert."""
        return self.failed == 0

    def __str__(self) -> str:
        tag = " [DRY RUN]" if self.dry_run else ""
        return (
            f"IndexResult{tag}: total={self.total}, indexed={self.indexed}, "
            f"skipped={self.skipped}, failed={self.failed}"
        )
