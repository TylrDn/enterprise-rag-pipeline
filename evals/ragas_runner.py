"""RAGAS evaluation runner with LangSmith dataset integration."""
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
from langsmith import Client
from orchestrator.graph import rag_graph
import os
from dotenv import load_dotenv

load_dotenv()

ls_client = Client()

# Sample eval dataset — replace with LangSmith dataset pull
EVAL_QUESTIONS = [
    {"question": "What is the company's Q4 revenue?", "ground_truth": "The Q4 revenue is reported in the financial summary."},
    {"question": "What are the product roadmap priorities?", "ground_truth": "The roadmap priorities are defined in the strategy document."},
    {"question": "Who are the key stakeholders for Project Alpha?", "ground_truth": "The stakeholders are listed in the project charter."},
]


def run_rag_for_eval(question: str) -> dict:
    initial_state = {
        "question": question,
        "rewritten_query": "",
        "documents": [],
        "graded_documents": [],
        "generation": "",
        "hallucination_score": 0.0,
        "answer_grade": "",
        "retry_count": 0,
        "source_types": [],
    }
    result = rag_graph.invoke(initial_state)
    return {
        "answer": result["generation"],
        "contexts": [d.page_content for d in result.get("graded_documents", [])],
    }


def run_ragas_eval():
    data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for item in EVAL_QUESTIONS:
        rag_result = run_rag_for_eval(item["question"])
        data["question"].append(item["question"])
        data["answer"].append(rag_result["answer"])
        data["contexts"].append(rag_result["contexts"])
        data["ground_truth"].append(item["ground_truth"])

    ds = Dataset.from_dict(data)
    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    print(result)
    return result


if __name__ == "__main__":
    run_ragas_eval()
