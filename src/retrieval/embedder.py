"""Embedder module — protocol definition, factory, and backward-compatible shim.

``EmbedderProtocol`` defines the structural interface for embedding providers.
``make_embedder`` is a factory that reads ``settings.embedding_provider`` and
returns the appropriate provider instance with lazy imports.
``OpenAIEmbedder`` is imported from ``src.retrieval.providers.embedder_openai``.
``Embedder`` is kept as a backward-compatibility alias.

New code should use ``make_embedder(settings)`` or import directly from the
provider modules under ``src.retrieval.providers``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from src.config import Settings


@runtime_checkable
class EmbedderProtocol(Protocol):
    """Structural interface that every embedding provider must satisfy."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per text."""
        ...

    @property
    def dimensions(self) -> int:
        """Number of dimensions in each embedding vector."""
        ...

    @property
    def provider_id(self) -> str:
        """Short identifier for the provider, e.g. ``"openai/text-embedding-3-small"``."""
        ...


def make_embedder(settings: Settings) -> EmbedderProtocol:
    """Return an embedder instance for the provider specified in *settings*.

    Provider modules are imported lazily inside this function so that importing
    ``src.retrieval.embedder`` does not pull in optional heavy dependencies
    (e.g. ``sentence-transformers``) unless they are actually needed.

    Raises:
        ValueError: If ``settings.embedding_provider`` is not a recognised value.
    """
    provider = settings.embedding_provider

    if provider == "openai":
        from src.retrieval.providers.embedder_openai import OpenAIEmbedder as _OpenAIEmbedder

        return _OpenAIEmbedder(settings)

    if provider == "sentence_transformers":
        from src.retrieval.providers.embedder_sentence_transformers import (
            DEFAULT_MODEL as _ST_DEFAULT_MODEL,
            SentenceTransformersEmbedder as _STEmbedder,
        )

        # Guard against accidentally passing an OpenAI model name (e.g. the default
        # "text-embedding-3-small") to the ST provider when the user has not
        # explicitly configured a sentence-transformers compatible model name.
        _OPENAI_MODEL_PREFIXES = ("text-embedding",)
        model_name = (
            _ST_DEFAULT_MODEL
            if any(settings.embedding_model.startswith(p) for p in _OPENAI_MODEL_PREFIXES)
            else settings.embedding_model
        )
        return _STEmbedder(model_name=model_name)

    valid = "openai, sentence_transformers, voyage, gemini, cohere"
    raise ValueError(
        f"Unknown embedding provider: {provider!r}. Valid providers are: {valid}"
    )


def __getattr__(name: str) -> object:
    """Lazy loader for backward-compatibility aliases.

    Provides ``Embedder`` and ``OpenAIEmbedder`` without importing
    ``src.retrieval.providers.embedder_openai`` at module load time.
    """
    if name in ("Embedder", "OpenAIEmbedder"):
        from src.retrieval.providers.embedder_openai import OpenAIEmbedder as _cls
        globals()["OpenAIEmbedder"] = _cls
        globals()["Embedder"] = _cls
        return _cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
