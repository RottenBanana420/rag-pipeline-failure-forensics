"""Voyage AI embedding provider.

``voyageai`` is imported lazily inside ``__init__`` so that this module can be
imported without the package being present.  Tests should patch
``voyageai.Client`` directly — Python's module cache means the patch is
visible to the inline import.
"""

from __future__ import annotations

from typing import cast

from src.config import Settings

BATCH_SIZE = 128  # Voyage's own docs recommend batches of 128 to avoid rate limits

DEFAULT_MODEL = "voyage-3.5"

# Dimension counts for known Voyage models (default output_dimension per model).
_MODEL_DIMENSIONS: dict[str, int] = {
    "voyage-4-large": 1024,
    "voyage-4": 1024,
    "voyage-4-lite": 1024,
    "voyage-3-large": 1024,
    "voyage-3.5": 1024,
    "voyage-3.5-lite": 1024,
    "voyage-code-3": 1024,
    "voyage-3": 1024,
    "voyage-2": 1024,
    "voyage-large-2": 1536,
    "voyage-code-2": 1536,
}


class VoyageEmbedder:
    """Embedding provider backed by the Voyage AI Embeddings API."""

    def __init__(self, settings: Settings) -> None:
        import voyageai  # lazy import — not at module level

        # voyageai's __init__.py doesn't declare `Client` in __all__, so mypy's strict
        # reexport check flags this despite it being the documented public API.
        self._client = voyageai.Client(  # type: ignore[attr-defined]
            api_key=settings.voyage_api_key
        )
        self._model = settings.embedding_model

    @property
    def dimensions(self) -> int:
        """Number of dimensions produced by the configured embedding model."""
        return _MODEL_DIMENSIONS.get(self._model, 1024)

    @property
    def provider_id(self) -> str:
        """Provider identifier including model name, e.g. ``"voyage/voyage-3.5"``."""
        return f"voyage/{self._model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per text, batching requests as needed."""
        if not texts:
            return []
        vectors: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            result = self._client.embed(batch, model=self._model, input_type="document")
            # We never set output_dtype, so Voyage always returns float embeddings;
            # the SDK's stub types this as float|int since output_dtype is configurable.
            vectors.extend(cast("list[list[float]]", result.embeddings))
        return vectors
