"""Shared pytest fixtures for the RAG pipeline test suite."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let the repo's real .env leak into Settings() in any test.

    pydantic-settings' dotenv source reads .env independently of os.environ,
    so a real .env (e.g. with a live ANTHROPIC_API_KEY or non-default judge
    models) would otherwise silently override field defaults in tests that
    only monkeypatch a handful of specific vars. Disabling env_file makes
    every test see field defaults plus exactly what it explicitly sets via
    monkeypatch.setenv — regardless of what's actually in .env.
    """
    from src.config import Settings

    monkeypatch.setitem(Settings.model_config, "env_file", None)


@pytest.fixture()
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip any real env vars so tests use field defaults or explicit overrides.

    NOTE: Tests must call Settings() directly (not import the module-level
    `settings` singleton) to get freshly validated values from monkeypatched env.
    """
    for var in [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "VOYAGE_API_KEY",
        "GEMINI_API_KEY",
        "COHERE_API_KEY",
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
        "CITATION_JUDGE_PROVIDER",
        "CITATION_JUDGE_MODEL",
        "CITATION_JUDGE_TEMPERATURE",
        "ANSWER_COMPLETENESS_JUDGE_PROVIDER",
        "ANSWER_COMPLETENESS_JUDGE_MODEL",
        "ANSWER_COMPLETENESS_JUDGE_TEMPERATURE",
        "CONFIDENCE_RETRIEVAL_WEIGHT",
        "CONFIDENCE_CITATION_WEIGHT",
        "CONFIDENCE_COMPLETENESS_WEIGHT",
        "TRACE_OUTPUT_DIR",
        "SQLITE_DB_PATH",
        "ROOT_CAUSE_JUDGE_PROVIDER",
        "ROOT_CAUSE_JUDGE_MODEL",
        "ROOT_CAUSE_JUDGE_TEMPERATURE",
        "ROOT_CAUSE_QUALITY_THRESHOLD",
    ]:
        monkeypatch.delenv(var, raising=False)
