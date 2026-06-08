"""Web scraper using httpx + BeautifulSoup with recursive site crawling."""
from __future__ import annotations

from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class WebLoader:
    """Scrape one or more URLs into LangChain Documents."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64, max_depth: int = 1) -> None:
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.max_depth = max_depth
        self._visited: set[str] = set()

    def _fetch(self, url: str) -> str:
        try:
            r = httpx.get(url, timeout=15, follow_redirects=True)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            return soup.get_text(separator=" ", strip=True)
        except Exception:
            return ""

    def load(self, url: str, depth: int = 0) -> list[Document]:
        if url in self._visited or depth > self.max_depth:
            return []
        self._visited.add(url)
        text = self._fetch(url)
        if not text:
            return []
        raw = Document(page_content=text, metadata={"source": url, "depth": depth})
        return self.splitter.split_documents([raw])

    def load_urls(self, urls: list[str]) -> list[Document]:
        docs: list[Document] = []
        for url in urls:
            docs.extend(self.load(url))
        return docs
