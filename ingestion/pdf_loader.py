"""PDF ingestion with PyMuPDF + recursive character text splitter."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64


class PDFLoader:
    """Load and chunk PDFs into LangChain Documents."""

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def load(self, path: str | Path) -> list[Document]:
        path = Path(path)
        doc = fitz.open(str(path))
        pages: list[Document] = []
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            if not text:
                continue
            pages.append(Document(
                page_content=text,
                metadata={
                    "source": str(path),
                    "page": i + 1,
                    "doc_hash": hashlib.md5(text.encode()).hexdigest(),
                },
            ))
        doc.close()
        return self.splitter.split_documents(pages)

    def load_directory(self, directory: str | Path) -> list[Document]:
        directory = Path(directory)
        docs: list[Document] = []
        for pdf in sorted(directory.glob("**/*.pdf")):
            docs.extend(self.load(pdf))
        return docs
