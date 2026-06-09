"""LangSmith evaluation harness for the RAG pipeline."""
from __future__ import annotations

import argparse
import logging
import os
from typing import Any

from dotenv import load_dotenv
from langsmith import Client
from langsmith.evaluation import evaluate
from langsmith.schemas import Example, Run

from orchestrator.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
DATASET_NAME = os.getenv("LANGSMITH_DATASET", "rag-pipeline-eval")

DEMO_QA = [
    {"input": {"question": "What is RAG?"}, "output": {"answer": "retrieval"}},
    {"input": {"question": "What is pgvector?"}, "output": {"answer": "postgres"}},
]

_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    """Return a lazily initialized pipeline for eval targets."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


def seed_dataset(client: Client) -> None:
    """Create the eval dataset in LangSmith when it does not already exist."""
    existing = {d.name for d in client.list_datasets()}
    if DATASET_NAME in existing:
        return
    dataset = client.create_dataset(DATASET_NAME)
    for example in DEMO_QA:
        client.create_example(
            inputs=example["input"],
            outputs=example["output"],
            dataset_id=dataset.id,
        )


def rag_target(inputs: dict[str, Any]) -> dict[str, Any]:
    """LangSmith target function wrapping the live pipeline."""
    question = inputs["question"]
    result = get_pipeline().answer_with_contexts(question)
    return {"answer": result["answer"], "contexts": result["contexts"]}


def relevance_evaluator(run: Run, example: Example) -> dict[str, Any]:
    """Score whether the predicted answer mentions the ground-truth keyword."""
    pred = (run.outputs or {}).get("answer", "")
    gt = (example.outputs or {}).get("answer", "")
    score = 1 if gt.lower() in pred.lower() else 0
    return {"key": "relevance", "score": score}


def run_eval(experiment_prefix: str = "rag-pipeline") -> Any:
    """Run LangSmith evaluation against the configured dataset."""
    client = Client(api_key=LANGSMITH_API_KEY)
    seed_dataset(client)
    results = evaluate(
        rag_target,
        data=DATASET_NAME,
        evaluators=[relevance_evaluator],
        experiment_prefix=experiment_prefix,
    )
    logger.info("LangSmith eval complete: %s", results)
    return results


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run LangSmith evaluation.")
    parser.add_argument(
        "--experiment-prefix",
        default="rag-pipeline",
        help="Prefix for the LangSmith experiment name.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    run_eval(experiment_prefix=args.experiment_prefix)
