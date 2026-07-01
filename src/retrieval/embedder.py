"""Embedder module — backward-compatible shim + protocol definition.

``OpenAIEmbedder`` is defined here (using a module-level ``OpenAI`` import so
that existing tests can patch ``src.retrieval.embedder.OpenAI``).
``Embedder`` is kept as an alias for backward compatibility.

New code should import from ``src.retrieval.providers.embedder_openai`` or
``src.retrieval.providers.embedder_sentence_transformers`` directly.
"""

from typing import Protocol, runtime_checkable

from openai import OpenAI

from src.config import Settings

BATCH_SIZE = 200

# Dimension counts for known OpenAI embedding models.
_MODEL_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


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


class OpenAIEmbedder:
    """OpenAI embedding provider (defined here to allow existing test patches to work).

    Existing tests patch ``src.retrieval.embedder.OpenAI`` — keeping ``OpenAI``
    imported at module level here preserves that behaviour.
    """

    def __init__(self, settings: Settings) -> None:
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.embedding_model

    @property
    def dimensions(self) -> int:
        """Number of dimensions produced by the configured embedding model."""
        return _MODEL_DIMENSIONS.get(self._model, 1536)

    @property
    def provider_id(self) -> str:
        """Provider identifier including model name, e.g. ``"openai/text-embedding-3-small"``."""
        return f"openai/{self._model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            response = self._client.embeddings.create(input=batch, model=self._model)
            vectors.extend(item.embedding for item in response.data)
        return vectors


# Backward-compatibility alias — existing code that imports ``Embedder`` keeps working.
Embedder = OpenAIEmbedder
