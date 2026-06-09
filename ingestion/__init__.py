"""Document loaders and async ingestion helpers."""
from __future__ import annotations

import asyncio

from langchain_core.documents import Document

from ingestion.pdf_loader import PDFLoader
from ingestion.slack_loader import SlackLoader
from ingestion.sql_loader import SQLLoader
from ingestion.web_loader import WebLoader

__all__ = [
    "PDFLoader",
    "SQLLoader",
    "WebLoader",
    "SlackLoader",
    "ingest_pdf",
    "ingest_url",
]


async def ingest_pdf(path: str) -> list[Document]:
    """Load a PDF asynchronously without blocking the event loop."""
    return await asyncio.to_thread(PDFLoader().load, path)


async def ingest_url(url: str) -> list[Document]:
    """Scrape a URL asynchronously without blocking the event loop."""
    return await asyncio.to_thread(WebLoader().load, url)
