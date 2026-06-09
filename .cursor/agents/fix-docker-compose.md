# Subagent: fix-docker-compose

**Invoke:** `/fix-docker-compose`
**Repo:** enterprise-rag-pipeline
**Task Ref:** TASKS.md §1.2
**Estimated Time:** 15 min

---

## Objective

Create a root-level `docker-compose.yml` that brings up the full RAG stack (API + vectorstore + Langfuse) from a single `docker compose up` command at the repo root. Update `README.md` to remove any `cd deploy/` instructions from the quick-start section.

---

## Context

- Existing compose file lives at `deploy/docker-compose.yml` (or similar); the root has none.
- The pattern to follow is `nvidia-nim-agent-toolkit`'s root `docker-compose.yml`: `env_file: .env`, build context `.`, reference to `deploy/Dockerfile`.
- Services needed: `api` (the FastAPI app), `qdrant` or `pgvector` (vectorstore), `langfuse` (observability).
- All secrets must come from `env_file: .env` — no hardcoded values.

---

## Step-by-Step Instructions

### Step 1 — Audit existing deploy/ compose

```bash
cat deploy/docker-compose.yml   # or find . -name docker-compose.yml
```

Note: existing ports, service names, volume mounts, health checks.

---

### Step 2 — Create root docker-compose.yml

Create `docker-compose.yml` at the repo root with the following structure:

```yaml
version: "3.9"

services:
  api:
    build:
      context: .
      dockerfile: deploy/Dockerfile
    ports:
      - "8000:8000"
    env_file: .env
    depends_on:
      vectorstore:
        condition: service_healthy
      langfuse:
        condition: service_started
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 30s
    restart: on-failure

  vectorstore:
    image: qdrant/qdrant:v1.9.0
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/healthz"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 15s

  langfuse:
    image: langfuse/langfuse:2
    ports:
      - "3000:3000"
    environment:
      DATABASE_URL: postgresql://langfuse:langfuse@langfuse_db:5432/langfuse
      NEXTAUTH_URL: http://localhost:3000
      NEXTAUTH_SECRET: ${LANGFUSE_NEXTAUTH_SECRET:-supersecretdev}
      SALT: ${LANGFUSE_SALT:-devonlysalt}
    depends_on:
      langfuse_db:
        condition: service_healthy
    restart: on-failure

  langfuse_db:
    image: postgres:15
    environment:
      POSTGRES_USER: langfuse
      POSTGRES_PASSWORD: langfuse
      POSTGRES_DB: langfuse
    volumes:
      - langfuse_db_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U langfuse"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  qdrant_data:
  langfuse_db_data:
```

**Key rules:**
- If the vectorstore backend is `pgvector` (not qdrant), swap in the `pgvector` image and update the health check URL accordingly.
- Do NOT hardcode any API keys. All NIM/OpenAI/Langfuse keys come from `env_file: .env`.
- If a `deploy/Dockerfile` does not exist, create a minimal one:
  ```dockerfile
  FROM python:3.11-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt
  COPY . .
  CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
  ```

---

### Step 3 — Verify compose config is valid

```bash
docker compose config
```

Must exit 0 with no warnings about missing variables (except GPU-optional ones).

---

### Step 4 — Update README.md quick-start section

Find the Getting Started / Quick Start section in `README.md`. Replace any instruction like:

```markdown
cd deploy && docker-compose up
```

With:

```markdown
# 1. Copy env template
cp .env.template .env
# Edit .env with your NIM_API_KEY, LANGFUSE_PUBLIC_KEY, etc.

# 2. Start the full stack
docker compose up

# 3. Verify
curl http://localhost:8000/health
```

---

### Step 5 — Add .env.template entries for compose secrets

Ensure `.env.template` includes:
```bash
# Docker Compose — Langfuse
LANGFUSE_NEXTAUTH_SECRET=<generate-random-32-char-string>
LANGFUSE_SALT=<generate-random-32-char-string>
```

---

## Acceptance Criteria

- [ ] `docker compose config` exits 0 from repo root
- [ ] `docker compose up` starts API on port 8000
- [ ] `curl http://localhost:8000/health` returns `{"status": "ok"}` within 60s of `docker compose up`
- [ ] No `cd deploy/` instruction remains in README quick-start section
- [ ] `.env.template` documents `LANGFUSE_NEXTAUTH_SECRET` and `LANGFUSE_SALT`
- [ ] `docker compose down -v` cleanly stops and removes all containers

---

## Do NOT

- Do NOT modify `deploy/docker-compose.yml` if it exists — leave it for k8s/CI use
- Do NOT commit a `.env` file — only `.env.template`
- Do NOT hardcode ports that conflict with `nvidia-nim-agent-toolkit` (port 8000 is shared; document this in README)
