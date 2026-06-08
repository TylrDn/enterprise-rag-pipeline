"""FastAPI RAG query gateway."""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from orchestrator.graph import rag_graph
from ingestion.pdf_loader import ingest_pdf
from ingestion.web_loader import ingest_url
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="enterprise-rag-pipeline", version="0.1.0")


class QueryRequest(BaseModel):
    question: str


class IngestPDFRequest(BaseModel):
    path: str


class IngestURLRequest(BaseModel):
    url: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    hallucination_score: float
    answer_grade: str


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        initial_state = {
            "question": request.question,
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
        sources = list({d.metadata.get("source", "unknown") for d in result.get("graded_documents", [])})
        return QueryResponse(
            question=request.question,
            answer=result["generation"],
            sources=sources,
            hallucination_score=result["hallucination_score"],
            answer_grade=result["answer_grade"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/pdf")
async def ingest_pdf_endpoint(request: IngestPDFRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(ingest_pdf, request.path)
    return {"status": "ingestion started", "path": request.path}


@app.post("/ingest/url")
async def ingest_url_endpoint(request: IngestURLRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(ingest_url, request.url)
    return {"status": "ingestion started", "url": request.url}


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
    )
