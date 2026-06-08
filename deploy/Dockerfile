# syntax=docker/dockerfile:1
# -----------------------------------------------------------------
# Enterprise RAG Pipeline — Production Dockerfile
# Multi-stage build: deps layer cached separately from app code.
# -----------------------------------------------------------------

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ---- deps stage ----
FROM base AS deps
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---- runtime stage ----
FROM deps AS runtime
COPY . .

# Non-root user for security
RUN addgroup --system rag && adduser --system --ingroup rag rag
USER rag

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
