"""Document relevance grader node."""
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from orchestrator.state import RAGState
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("CHAT_MODEL", "meta/llama3-70b-instruct"),
    base_url=os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0,
)

GRADE_PROMPT = """Is the following document relevant to answering the question?
Answer only 'yes' or 'no'.

Question: {question}
Document: {document}"""


def grade_document(question: str, doc: Document) -> bool:
    prompt = GRADE_PROMPT.format(question=question, document=doc.page_content[:800])
    result = llm.invoke(prompt).content.strip().lower()
    return result.startswith("yes")


def grade_documents(state: RAGState) -> RAGState:
    question = state.get("rewritten_query") or state["question"]
    docs = state["documents"]
    graded = [doc for doc in docs if grade_document(question, doc)]
    return {
        **state,
        "graded_documents": graded,
        "retry_count": state.get("retry_count", 0) + (0 if graded else 1),
    }
