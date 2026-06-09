"""Unit tests for document chunker."""
from langchain_core.documents import Document

from embeddings.chunker import markdown_chunk, recursive_chunk


def test_markdown_chunk_splits_on_headers_and_preserves_metadata():
    doc = Document(
        page_content="# Title\nintro text\n## Section\nbody content here",
        metadata={"source": "guide.md"},
    )
    chunks = markdown_chunk([doc])
    assert len(chunks) >= 1
    assert all(c.metadata.get("source") == "guide.md" for c in chunks)


def test_recursive_chunk_splits_long_doc():
    doc = Document(page_content="word " * 300, metadata={"source": "test"})
    chunks = recursive_chunk([doc], chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content) <= 150  # allow some overlap headroom


def test_recursive_chunk_preserves_metadata():
    doc = Document(
        page_content="Some content here.", metadata={"source": "test.pdf", "type": "pdf"}
    )
    chunks = recursive_chunk([doc])
    assert all(c.metadata["source"] == "test.pdf" for c in chunks)
