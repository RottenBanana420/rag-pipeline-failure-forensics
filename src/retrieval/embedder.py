"""Embedder module — protocol definition and backward-compatible shim.

``EmbedderProtocol`` defines the structural interface for embedding providers.
``OpenAIEmbedder`` is imported from ``src.retrieval.providers.embedder_openai``.
``Embedder`` is kept as a backward-compatibility alias.

New code should import from ``src.retrieval.providers.embedder_openai`` or
``src.retrieval.providers.embedder_sentence_transformers`` directly.
"""

from typing import Protocol, runtime_checkable

from src.retrieval.providers.embedder_openai import OpenAIEmbedder


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


# Backward-compatibility alias — existing code that imports ``Embedder`` keeps working.
Embedder = OpenAIEmbedder
