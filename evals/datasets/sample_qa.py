"""Sample ground-truth QA pairs for RAGAS and LangSmith evals."""

QA_PAIRS = [
    {
        "question": "What is retrieval-augmented generation?",
        "ground_truth": "Retrieval-augmented generation (RAG) is a technique that combines a retrieval system with a language model to ground answers in retrieved documents.",
    },
    {
        "question": "What are the key metrics in RAGAS?",
        "ground_truth": "RAGAS measures faithfulness, answer relevancy, context recall, and context precision.",
    },
    {
        "question": "What is pgvector?",
        "ground_truth": "pgvector is a PostgreSQL extension that adds vector similarity search capabilities, commonly used as a vector store for RAG pipelines.",
    },
]
