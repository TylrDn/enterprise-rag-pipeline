"""Web page ingestion via httpx + BeautifulSoup."""
import httpx
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from embeddings.chunker import recursive_chunk
from vectorstore.pgvector_store import PGVectorStore
from typing import List


def load_url(url: str) -> List[Document]:
    response = httpx.get(url, follow_redirects=True, timeout=30.0)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return [Document(page_content=text, metadata={"source": url, "type": "web"})]


def ingest_url(url: str, store: PGVectorStore = None) -> List[str]:
    docs = load_url(url)
    chunks = recursive_chunk(docs)
    vs = store or PGVectorStore()
    ids = vs.add_documents(chunks)
    print(f"Ingested {len(chunks)} chunks from {url}")
    return ids
