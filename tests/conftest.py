"""Shared pytest fixtures for the RAG pipeline test suite."""

from __future__ import annotations

import pytest


@pytest.fixture()
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip any real env vars so tests use field defaults or explicit overrides.

    NOTE: Tests must call Settings() directly (not import the module-level
    `settings` singleton) to get freshly validated values from monkeypatched env.
    """
    for var in [
        "OPENAI_API_KEY",
        "EMBEDDING_MODEL",
        "CHROMA_PERSIST_DIR",
        "DENSE_TOP_K",
        "SPARSE_TOP_K",
        "RERANK_CANDIDATE_POOL",
        "RERANK_TOP_N",
        "DENSE_WEIGHT",
        "SPARSE_WEIGHT",
        "RERANKING_ENABLED",
        "RERANKER_PROVIDER",
        "RERANKER_MODEL",
        "RERANKER_DEVICE",
        "DEDUP_THRESHOLD",
        "LOG_LEVEL",
        "CHUNK_STRATEGY",
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "SEMANTIC_BREAKPOINT_PERCENTILE",
    ]:
        monkeypatch.delenv(var, raising=False)
