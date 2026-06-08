"""pgvector store wrapper."""
from langchain_postgres import PGVector
from langchain_core.documents import Document
from embeddings.nim_embedder import NIMEmbedder
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()


class PGVectorStore:
    def __init__(self, collection: str = None):
        self.collection = collection or os.getenv("PGVECTOR_COLLECTION", "documents")
        self.embedder = NIMEmbedder()
        self.store = PGVector(
            embeddings=self.embedder,
            collection_name=self.collection,
            connection=os.getenv("DATABASE_URL"),
            use_jsonb=True,
        )

    def add_documents(self, docs: List[Document]) -> List[str]:
        return self.store.add_documents(docs)

    def similarity_search(self, query: str, k: int = 6) -> List[Document]:
        return self.store.similarity_search(query, k=k)

    def similarity_search_with_score(self, query: str, k: int = 6):
        return self.store.similarity_search_with_relevance_scores(query, k=k)

    def as_retriever(self, k: int = 6):
        return self.store.as_retriever(search_kwargs={"k": k})

    def delete_collection(self):
        self.store.delete_collection()
