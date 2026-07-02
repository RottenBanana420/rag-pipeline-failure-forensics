"""Cohere embedding provider.

``cohere`` is imported lazily inside ``__init__`` so that this module can be
imported without the package being present.  Tests should patch
``cohere.ClientV2`` directly — Python's module cache means the patch is
visible to the inline import.
"""

from __future__ import annotations

from src.config import Settings

BATCH_SIZE = 96  # Cohere's documented max texts per embed() call

DEFAULT_MODEL = "embed-v4.0"

# Dimension counts for known Cohere embedding models.
_MODEL_DIMENSIONS: dict[str, int] = {
    "embed-v4.0": 1024,
    "embed-english-v3.0": 1024,
    "embed-multilingual-v3.0": 1024,
    "embed-english-light-v3.0": 384,
    "embed-multilingual-light-v3.0": 384,
}


class CohereEmbedder:
    """Embedding provider backed by the Cohere Embeddings API."""

    def __init__(self, settings: Settings) -> None:
        import cohere  # lazy import — not at module level

        self._client = cohere.ClientV2(api_key=settings.cohere_api_key)
        self._model = settings.embedding_model

    @property
    def dimensions(self) -> int:
        """Number of dimensions produced by the configured embedding model."""
        return _MODEL_DIMENSIONS.get(self._model, 1024)

    @property
    def provider_id(self) -> str:
        """Provider identifier including model name, e.g. ``"cohere/embed-v4.0"``."""
        return f"cohere/{self._model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per text, batching requests as needed."""
        if not texts:
            return []
        vectors: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            response = self._client.v2.embed(
                model=self._model,
                input_type="search_document",
                texts=batch,
                embedding_types=["float"],
            )
            vectors.extend(response.embeddings.float_)
        return vectors
