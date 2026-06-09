"""Vector store abstraction and backend factory."""
from vectorstore.base import VectorStoreBase
from vectorstore.factory import get_vector_store

__all__ = ["VectorStoreBase", "get_vector_store"]
