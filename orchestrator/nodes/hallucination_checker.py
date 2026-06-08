"""Hallucination grounding checker node."""
from langchain_openai import ChatOpenAI
from orchestrator.state import RAGState
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("CHAT_MODEL", "meta/llama3-70b-instruct"),
    base_url=os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0,
)

HALLUCINATION_PROMPT = """You are a factuality checker. Score how well the answer is grounded in the provided context.
Respond with only a number between 0.0 (completely hallucinated) and 1.0 (fully grounded).

Context:
{context}

Answer: {answer}

Grounding score:"""


def check_hallucination(state: RAGState) -> RAGState:
    docs = state.get("graded_documents") or state.get("documents", [])
    context = "\n\n".join(d.page_content[:600] for d in docs)
    answer = state["generation"]
    prompt = HALLUCINATION_PROMPT.format(context=context, answer=answer)
    try:
        score = float(llm.invoke(prompt).content.strip())
    except ValueError:
        score = 0.5
    answer_grade = "supported" if score >= 0.6 else "unsupported"
    return {**state, "hallucination_score": score, "answer_grade": answer_grade}
