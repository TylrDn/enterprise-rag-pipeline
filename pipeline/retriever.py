"""Hybrid retriever: BM25 + dense vector with cross-encoder reranking."""
from langchain.schema import Document, BaseRetriever
from langchain.embeddings.base import Embeddings
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from typing import List
import logging

logger = logging.getLogger(__name__)


def build_hybrid_retriever(
    docs: List[Document],
    dense_retriever: BaseRetriever,
    top_k: int = 10,
    rerank_top_n: int = 4,
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> ContextualCompressionRetriever:
    """BM25 + dense ensemble → cross-encoder reranker pipeline."""
    bm25 = BM25Retriever.from_documents(docs, k=top_k)
    ensemble = EnsembleRetriever(
        retrievers=[bm25, dense_retriever],
        weights=[0.4, 0.6],
    )
    encoder = HuggingFaceCrossEncoder(model_name=reranker_model)
    compressor = CrossEncoderReranker(model=encoder, top_n=rerank_top_n)
    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble)
