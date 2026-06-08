"""RAG agent state definition."""
from typing import TypedDict, List, Optional
from langchain_core.documents import Document


class RAGState(TypedDict):
    question: str                        # original user question
    rewritten_query: str                 # query after rewriting
    documents: List[Document]            # retrieved documents
    graded_documents: List[Document]     # documents passing relevance grade
    generation: str                      # LLM generated answer
    hallucination_score: float           # faithfulness score
    answer_grade: str                    # "supported" | "unsupported" | "not_useful"
    retry_count: int                     # number of retries
    source_types: List[str]              # which sources were used
