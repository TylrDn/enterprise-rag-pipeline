"""PDF ingestion: PyMuPDF → chunked LangChain Documents."""
import fitz  # PyMuPDF
from pathlib import Path
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
import logging

logger = logging.getLogger(__name__)


def load_pdf(path: str | Path, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """Load a PDF, extract text with metadata, and split into chunks."""
    doc = fitz.open(str(path))
    raw_docs: List[Document] = []

    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            raw_docs.append(Document(
                page_content=text,
                metadata={
                    "source": str(path),
                    "page": page_num + 1,
                    "total_pages": len(doc),
                    "loader": "pdf",
                }
            ))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)
    logger.info(f"pdf_loader: {len(chunks)} chunks from {path}")
    return chunks
