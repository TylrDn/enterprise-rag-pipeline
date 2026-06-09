"""Sample ground-truth QA pairs for RAGAS and LangSmith evals."""

QA_PAIRS = [
    {
        "question": "What is retrieval-augmented generation?",
        "ground_truth": (
            "Retrieval-augmented generation (RAG) combines retrieval with a language "
            "model to ground answers in retrieved documents."
        ),
    },
    {
        "question": "What are the key metrics in RAGAS?",
        "ground_truth": (
            "RAGAS measures faithfulness, answer relevancy, context recall, "
            "and context precision."
        ),
    },
    {
        "question": "What is pgvector?",
        "ground_truth": (
            "pgvector is a PostgreSQL extension for vector similarity search, "
            "commonly used as a vector store for RAG pipelines."
        ),
    },
]
