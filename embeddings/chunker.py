"""Document chunking strategies."""
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()


def recursive_chunk(docs: List[Document], chunk_size: int = None, chunk_overlap: int = None) -> List[Document]:
    size = chunk_size or int(os.getenv("CHUNK_SIZE", "512"))
    overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", "64"))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def markdown_chunk(docs: List[Document]) -> List[Document]:
    headers = [("#", "h1"), ("##", "h2"), ("###", "h3")]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
    chunks = []
    for doc in docs:
        splits = splitter.split_text(doc.page_content)
        for s in splits:
            s.metadata.update(doc.metadata)
        chunks.extend(splits)
    return chunks
