"""FAISS in-memory vector store wrapper."""
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from embeddings.nim_embedder import NIMEmbedder
from typing import List
import os


class FAISSStore:
    def __init__(self):
        self.embedder = NIMEmbedder()
        self.store = None

    def build(self, docs: List[Document]):
        self.store = FAISS.from_documents(docs, self.embedder)
        return self

    def load(self, path: str):
        self.store = FAISS.load_local(path, self.embedder, allow_dangerous_deserialization=True)
        return self

    def save(self, path: str):
        if self.store:
            self.store.save_local(path)

    def similarity_search(self, query: str, k: int = 6) -> List[Document]:
        return self.store.similarity_search(query, k=k)

    def as_retriever(self, k: int = 6):
        return self.store.as_retriever(search_kwargs={"k": k})
