"""LangSmith evaluation harness — dataset upload + evaluator config."""
from langsmith import Client
from langsmith.evaluation import evaluate as ls_evaluate
from typing import List, Dict, Any, Callable
import os


def upload_dataset(
    name: str,
    examples: List[Dict[str, Any]],
    description: str = "",
) -> str:
    """Upload QA pairs to LangSmith as a named dataset. Returns dataset ID."""
    client = Client(api_key=os.environ["LANGSMITH_API_KEY"])
    dataset = client.create_dataset(dataset_name=name, description=description)
    client.create_examples(
        inputs=[{"question": e["question"]} for e in examples],
        outputs=[{"answer": e["answer"]} for e in examples],
        dataset_id=dataset.id,
    )
    print(f"Uploaded {len(examples)} examples to dataset '{name}' ({dataset.id})")
    return str(dataset.id)


def run_langsmith_eval(
    target_fn: Callable,
    dataset_name: str,
    experiment_prefix: str = "rag-eval",
) -> Dict[str, Any]:
    """Run LangSmith evaluation against an existing dataset."""
    results = ls_evaluate(
        target_fn,
        data=dataset_name,
        experiment_prefix=experiment_prefix,
    )
    print(f"LangSmith eval complete: {results}")
    return results
