"""RAGAS evaluation: faithfulness, answer_relevancy, context_recall, context_precision."""
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from datasets import Dataset
from typing import List, Dict, Any
import json
import pathlib
import datetime


def run_ragas_eval(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ground_truths: List[str],
    report_dir: str = "evals/reports",
) -> Dict[str, Any]:
    """Run RAGAS evaluation suite and write JSON report."""
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )
    scores = result.to_pandas().mean().to_dict()

    pathlib.Path(report_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_path = f"{report_dir}/ragas_{ts}.json"
    with open(report_path, "w") as f:
        json.dump({"scores": scores, "timestamp": ts}, f, indent=2)

    print(f"RAGAS scores: {scores}")
    print(f"Report saved: {report_path}")
    return scores
