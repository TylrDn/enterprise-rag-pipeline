"""PDF ingestion using pdfplumber + unstructured."""
from langchain_community.document_loaders import PDFPlumberLoader, UnstructuredPDFLoader
from langchain_core.documents import Document
from embeddings.chunker import recursive_chunk
from vectorstore.pgvector_store import PGVectorStore
from typing import List
import os


def load_pdf(path: str, strategy: str = "pdfplumber") -> List[Document]:
    if strategy == "unstructured":
        loader = UnstructuredPDFLoader(path, mode="elements")
    else:
        loader = PDFPlumberLoader(path)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = os.path.basename(path)
        doc.metadata["type"] = "pdf"
    return docs


def ingest_pdf(path: str, store: PGVectorStore = None) -> List[str]:
    docs = load_pdf(path)
    chunks = recursive_chunk(docs)
    vs = store or PGVectorStore()
    ids = vs.add_documents(chunks)
    print(f"Ingested {len(chunks)} chunks from {path}")
    return ids
