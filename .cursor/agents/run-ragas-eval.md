---
name: run-ragas-eval
description: Invoke this agent to execute a RAGAS evaluation run against the pipeline — either to benchmark a new model or embedding change, to investigate a drop in retrieval quality, or to generate evaluation results for a dataset file in evals/datasets/.
model: inherit
readonly: false
---

# Run RAGAS Evaluation — enterprise-rag-pipeline

## Objective

Execute a complete RAGAS evaluation run: load a QA dataset, run the RAG pipeline against it, compute RAGAS metrics (faithfulness, answer relevancy, context precision, context recall), output results as JSON, and optionally upload results to LangSmith. Identify any metrics below threshold and provide actionable recommendations.

## Context

RAGAS evaluates four key metrics on RAG pipelines:
- **Faithfulness**: Is the generated answer grounded in the retrieved context? (Target: ≥ 0.85)
- **Answer Relevancy**: Does the answer actually address the question? (Target: ≥ 0.80)
- **Context Precision**: Are the retrieved chunks relevant to the question? (Target: ≥ 0.75)
- **Context Recall**: Do the retrieved chunks cover the expected answer? (Target: ≥ 0.70)

## Pre-flight Checks

1. Confirm RAGAS is installed: `pip show ragas` — if not, `pip install ragas`.
2. Confirm evaluation dataset exists: look in `evals/datasets/` for `.json` or `.jsonl` files.
3. Confirm environment variables are set: `NIM_BASE_URL`, `NIM_API_KEY`, `NIM_EMBEDDING_MODEL`, `NIM_GENERATOR_MODEL`, `VECTORSTORE_BACKEND`.
4. Confirm vector store is populated (or will be populated as part of this eval run).
5. If uploading to LangSmith: confirm `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT`.

## Step-by-Step Instructions

### Step 1 — Identify the Dataset

Read the contents of `evals/datasets/`. List all available dataset files. If no dataset files exist, create a minimal test dataset:

```python
# evals/datasets/sample_qa.json
[
  {
    "question": "What is NVIDIA NIM?",
    "ground_truth": "NVIDIA NIM is a set of microservices for deploying AI models with an OpenAI-compatible API.",
    "contexts": []  # Will be filled by the retriever
  },
  {
    "question": "How does the CRAG orchestrator improve retrieval quality?",
    "ground_truth": "CRAG grades retrieved documents for relevance and triggers a web search fallback when documents are insufficient.",
    "contexts": []
  }
]
```

### Step 2 — Run the Evaluation via `evals/ragas_runner.py`

```bash
python evals/ragas_runner.py \
  --dataset evals/datasets/sample_qa.json \
  --output-file evals/results/ragas_results_$(date +%Y%m%d_%H%M%S).json \
  --metrics faithfulness,answer_relevancy,context_precision,context_recall
```

If `ragas_runner.py` does not accept these arguments, read the file and determine the correct invocation syntax, then run accordingly.

### Step 3 — Read and Interpret Results

After the run completes, read the output JSON file. Parse the results:

```python
import json

with open("evals/results/ragas_results_TIMESTAMP.json") as f:
    results = json.load(f)

thresholds = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.80,
    "context_precision": 0.75,
    "context_recall": 0.70,
}

for metric, threshold in thresholds.items():
    score = results.get(metric, results.get("scores", {}).get(metric))
    status = "PASS" if score >= threshold else "FAIL"
    print(f"{metric}: {score:.3f} (threshold {threshold}) — {status}")
```

### Step 4 — Diagnose Failures

For each metric below threshold, apply the corresponding diagnostic:

#### Faithfulness < 0.85
- Symptom: LLM is generating answers not grounded in retrieved context (hallucination).
- Investigation: Sample 5 failing question/answer/context triplets. Is the LLM ignoring the context?
- Remediation:
  - Tighten the system prompt in `pipeline/generator.py`: add "Answer ONLY using the provided context. If the context does not contain the answer, say 'I don't know'."
  - Reduce `temperature` to 0 on the generator `ChatOpenAI`.
  - Check if context chunks are too long — truncate to fit within the model's context window.

#### Answer Relevancy < 0.80
- Symptom: Answers are technically grounded but don't address the user's question.
- Investigation: Are the retrieved chunks relevant? Or is the generation step losing the question intent?
- Remediation:
  - Review the RAG chain prompt template in `pipeline/generator.py`.
  - Increase `top_k` in `retriever/hybrid_retriever.py`.
  - Check the hybrid retrieval fusion weights.

#### Context Precision < 0.75
- Symptom: Retrieved chunks include too many irrelevant passages.
- Investigation: Sample retrieval results for failing questions. Are irrelevant chunks being returned?
- Remediation:
  - Reduce `top_k` to return fewer but more precise chunks.
  - Review the CRAG grading threshold — is the grade LLM too lenient?
  - Check embedding model — try a different `NIM_EMBEDDING_MODEL`.

#### Context Recall < 0.70
- Symptom: Relevant information exists in the corpus but is not being retrieved.
- Investigation: Does the vector store contain the relevant documents? Check ingestion logs.
- Remediation:
  - Re-index with smaller chunk size (`CHUNK_SIZE=256`).
  - Verify hybrid retrieval is using both dense and sparse components.
  - Check if the web search fallback in CRAG is being triggered when needed.

### Step 5 — Optional: Upload to LangSmith

If `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` are set:

```bash
python evals/langsmith_eval.py \
  --results-file evals/results/ragas_results_TIMESTAMP.json \
  --project-name enterprise-rag-pipeline-eval
```

### Step 6 — Report

Output a markdown summary:

```markdown
## RAGAS Evaluation Report

**Date:** YYYY-MM-DD HH:MM
**Dataset:** evals/datasets/FILENAME.json
**Questions evaluated:** N
**Results file:** evals/results/ragas_results_TIMESTAMP.json

| Metric | Score | Threshold | Status |
|---|---|---|---|
| Faithfulness | X.XXX | 0.85 | PASS/FAIL |
| Answer Relevancy | X.XXX | 0.80 | PASS/FAIL |
| Context Precision | X.XXX | 0.75 | PASS/FAIL |
| Context Recall | X.XXX | 0.70 | PASS/FAIL |

### Recommendations
- [List specific actionable recommendations based on failures]
- [Or: "All metrics above threshold — no action required."]
```

## Acceptance Criteria

- [ ] RAGAS evaluation completes without Python errors.
- [ ] Output JSON file is written to `evals/results/` with a timestamp in the filename.
- [ ] All four metrics are present in the output: faithfulness, answer_relevancy, context_precision, context_recall.
- [ ] Metrics are compared against thresholds and a PASS/FAIL determination is made for each.
- [ ] For any FAIL, at least one specific remediation action is identified.
- [ ] A markdown summary report is produced (output to terminal or saved to `evals/results/report_TIMESTAMP.md`).
- [ ] If `LANGSMITH_API_KEY` is set, results are uploaded to LangSmith.
- [ ] No eval imports appear in `pipeline/` or `orchestrator/` modules (eval isolation maintained).
