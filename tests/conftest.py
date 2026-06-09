"""Shared pytest fixtures for the enterprise-rag-pipeline test suite."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide dummy credentials so module-level clients construct without network."""
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.setenv("NIM_API_KEY", "test-key")
    monkeypatch.setenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
    monkeypatch.setenv("PGVECTOR_URL", "postgresql+psycopg://u:p@localhost/test")
    monkeypatch.setenv("VECTORSTORE_BACKEND", "pgvector")


@pytest.fixture()
def mock_langfuse() -> MagicMock:
    """Patch Langfuse handler creation so tests never open network connections."""
    handler = MagicMock()
    with patch("core.observability.get_langfuse_handler", return_value=handler), patch(
        "core.observability.get_callbacks", return_value=[handler]
    ):
        yield handler
