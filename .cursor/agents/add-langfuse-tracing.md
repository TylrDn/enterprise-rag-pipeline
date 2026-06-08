---
name: add-langfuse-tracing
description: Invoke this agent when Langfuse tracing is missing from pipeline/generator.py or any other LLM call site in the enterprise-rag-pipeline — specifically when CallbackHandler is not passed to ChatOpenAI or chain invocations in the generation or orchestrator layers.
model: inherit
readonly: false
---

# Add Langfuse Tracing to enterprise-rag-pipeline

## Objective

Instrument all LLM call paths in this repository with Langfuse tracing via `langfuse.callback.CallbackHandler`. The primary gap is in `pipeline/generator.py`. Secondary gaps may exist in `orchestrator/` CRAG nodes. When complete, every language model invocation in the pipeline will emit a trace to Langfuse, and the Langfuse handler factory will live in a shared observability module.

## Context

This repo uses `langchain_openai.ChatOpenAI` for the RAG generation step in `pipeline/generator.py` and for document grading in the CRAG orchestrator. Currently, `CallbackHandler` is not passed to these LLM calls. This also creates a gap in the eval loop — traces won't appear in Langfuse for debugging retrieval quality.

Note: LangSmith tracing is already partially configured via `LANGCHAIN_TRACING_V2` env vars. Langfuse is additive — both run concurrently.

## Files to Touch

1. `pipeline/generator.py` — Primary fix. Add `get_langfuse_handler()` and inject into `ChatOpenAI` and all `chain.invoke()` calls.
2. `orchestrator/` (all node files in the CRAG graph) — Inject handler into LLM calls in grade and generate nodes.
3. `.env.template` — Add Langfuse environment variables.
4. `requirements.txt` — Add `langfuse>=2.0.0`.
5. `tests/conftest.py` — Add `mock_langfuse` fixture.
6. `pyproject.toml` — Add langfuse to `[project.dependencies]` if this file is the canonical deps source.

Optionally create:
- `core/observability.py` — Shared observability utilities (handler factory, LangSmith setup). Avoids duplication across `pipeline/generator.py` and orchestrator files.

## Step-by-Step Instructions

### Step 1 — Create `core/observability.py` (Recommended)

Create a new file `core/observability.py`:

```python
"""Shared observability utilities for enterprise-rag-pipeline.

Provides Langfuse and LangSmith callback handler factories used across
the pipeline, orchestrator, and evaluation layers.
"""

import logging
import os

from langfuse.callback import CallbackHandler

logger = logging.getLogger(__name__)


def get_langfuse_handler() -> CallbackHandler:
    """Return a configured Langfuse CallbackHandler.

    Reads the following environment variables:
        LANGFUSE_PUBLIC_KEY: Langfuse project public key.
        LANGFUSE_SECRET_KEY: Langfuse project secret key.
        LANGFUSE_HOST: Langfuse host URL (default: https://cloud.langfuse.com).

    Returns:
        CallbackHandler: Configured handler. Operates in no-op mode if
            credentials are missing (logs a warning).
    """
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")
    host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not public_key or not secret_key:
        logger.warning(
            "LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY not configured. "
            "Langfuse tracing will be disabled for this session."
        )

    return CallbackHandler(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
    )


def get_callbacks() -> list:
    """Return a list of all active callback handlers.

    Combines Langfuse handler with any other handlers that should be
    applied globally (e.g., LangSmith is configured via env vars and
    does not require an explicit callback object).

    Returns:
        list: List containing the Langfuse CallbackHandler.
    """
    return [get_langfuse_handler()]
```

### Step 2 — Update `pipeline/generator.py`

Read the current `pipeline/generator.py` to understand its structure. Then apply the following changes:

**Add import:**
```python
from core.observability import get_callbacks
```

**Find the `ChatOpenAI` construction** and update it:
```python
# Before:
llm = ChatOpenAI(
    model=os.environ["NIM_GENERATOR_MODEL"],
    base_url=os.environ["NIM_BASE_URL"],
    api_key=os.environ["NIM_API_KEY"],
    temperature=0,
)

# After:
llm = ChatOpenAI(
    model=os.environ["NIM_GENERATOR_MODEL"],
    base_url=os.environ["NIM_BASE_URL"],
    api_key=os.environ["NIM_API_KEY"],
    temperature=0,
    callbacks=get_callbacks(),
)
```

**Find any `chain.invoke()` or `llm.ainvoke()` calls** and update:
```python
# Before:
result = await chain.ainvoke({"context": context, "question": question})

# After:
result = await chain.ainvoke(
    {"context": context, "question": question},
    config={"callbacks": get_callbacks()},
)
```

### Step 3 — Update CRAG Orchestrator Nodes

Read all files in `orchestrator/`. For any file that:
1. Constructs a `ChatOpenAI` instance — add `callbacks=get_callbacks()`.
2. Calls `llm.ainvoke()` or `chain.ainvoke()` — add `config={"callbacks": get_callbacks()}`.

The CRAG pattern typically has:
- A document grading node (LLM call to score relevance)
- A generation node (LLM call to produce answer)
- Potentially a question rewriting node (LLM call)

All of these must receive the callback handler.

Add to each orchestrator file:
```python
from core.observability import get_callbacks
```

### Step 4 — Update `.env.template`

Add to `.env.template`:
```bash
# Langfuse Observability (https://langfuse.com)
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key_here
LANGFUSE_SECRET_KEY=your_langfuse_secret_key_here
LANGFUSE_HOST=https://cloud.langfuse.com

# LangSmith (Optional — runs concurrently with Langfuse)
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=your_langsmith_api_key_here
# LANGCHAIN_PROJECT=enterprise-rag-pipeline
```

### Step 5 — Update `requirements.txt`

Add:
```
langfuse>=2.0.0
```

Verify no version conflicts with existing `langchain-core` or `langchain-openai` pins.

### Step 6 — Add `mock_langfuse` fixture to `tests/conftest.py`

```python
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=False)
def mock_langfuse():
    """Patch Langfuse CallbackHandler to prevent outbound network calls in unit tests.

    Apply this fixture to any test that creates a ChatOpenAI instance or
    calls pipeline/orchestrator functions that trigger LLM construction.
    """
    with patch("langfuse.callback.CallbackHandler") as mock_handler_cls:
        mock_instance = MagicMock()
        mock_handler_cls.return_value = mock_instance
        yield mock_instance


@pytest.fixture(autouse=False)
def mock_observability(mocker):
    """Patch core.observability.get_callbacks to return an empty list in unit tests."""
    return mocker.patch("core.observability.get_callbacks", return_value=[])
```

Apply `mock_observability` fixture to all unit tests that exercise generator or orchestrator code.

## Acceptance Criteria

- [ ] `core/observability.py` exists with `get_langfuse_handler()` and `get_callbacks()` functions.
- [ ] `pipeline/generator.py` imports `get_callbacks` and passes it to every `ChatOpenAI` constructor and every `chain.ainvoke()` call.
- [ ] All CRAG orchestrator nodes that construct LLMs or call chains include `callbacks=get_callbacks()` or `config={"callbacks": get_callbacks()}`.
- [ ] `.env.template` contains `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`.
- [ ] `langfuse>=2.0.0` is in `requirements.txt`.
- [ ] `tests/conftest.py` has `mock_langfuse` and `mock_observability` fixtures.
- [ ] `pytest tests/ -m "not integration"` passes with no new failures.
- [ ] `ruff check . --fix && mypy .` pass clean.
- [ ] Running `grep -rn "ChatOpenAI(" . --include="*.py"` shows every result has `callbacks=` present.
- [ ] No duplication of `get_langfuse_handler()` — all files import from `core/observability.py`.
