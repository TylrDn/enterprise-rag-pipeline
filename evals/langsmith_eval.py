"""LangSmith evaluation harness for the RAG pipeline."""
from __future__ import annotations

import os
from typing import Any

from langsmith import Client
from langsmith.evaluation import evaluate
from langsmith.schemas import Run, Example

LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
DATASET_NAME = os.getenv("LANGSMITH_DATASET", "rag-pipeline-eval")

DEMO_QA = [
    {"input": {"question": "What is RAG?"}, "output": {"answer": "retrieval"}},
    {"input": {"question": "What is pgvector?"}, "output": {"answer": "postgres"}},
]


def seed_dataset(client: Client) -> None:
    existing = {d.name for d in client.list_datasets()}
    if DATASET_NAME in existing:
        return
    ds = client.create_dataset(DATASET_NAME)
    for ex in DEMO_QA:
        client.create_example(inputs=ex["input"], outputs=ex["output"], dataset_id=ds.id)


def rag_target(inputs: dict[str, Any]) -> dict[str, Any]:
    # Wire your pipeline here
    return {"answer": "placeholder"}


def relevance_evaluator(run: Run, example: Example) -> dict:
    pred = (run.outputs or {}).get("answer", "")
    gt = (example.outputs or {}).get("answer", "")
    score = 1 if gt.lower() in pred.lower() else 0
    return {"key": "relevance", "score": score}


def run_eval() -> None:
    client = Client(api_key=LANGSMITH_API_KEY)
    seed_dataset(client)
    results = evaluate(
        rag_target,
        data=DATASET_NAME,
        evaluators=[relevance_evaluator],
        experiment_prefix="rag-pipeline",
    )
    for r in results:
        print(r)


if __name__ == "__main__":
    run_eval()
