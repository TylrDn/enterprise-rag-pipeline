"""Live integration tests — require pgvector and NIM; skipped in default CI."""

from __future__ import annotations

import os

import pytest


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION") != "1",
    reason="Set RUN_INTEGRATION=1 to run live service tests.",
)
def test_nim_models_reachable() -> None:
    """Smoke test that NIM credentials are configured for integration runs."""
    assert os.getenv("NVIDIA_API_KEY") or os.getenv("NIM_API_KEY")
