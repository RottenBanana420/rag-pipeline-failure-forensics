"""OpenAI embedding provider.

``openai`` is imported lazily inside the constructor so that this module can be
imported without the package being present.  Tests should patch
``src.retrieval.providers.embedder_openai.OpenAI`` — the name is bound once on
the first instantiation and cached as a module-level name so subsequent
patches also work.
"""

from __future__ import annotations

from src.config import Settings

BATCH_SIZE = 200

# Dimension counts for known OpenAI embedding models.
_MODEL_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

# Module-level placeholder — populated on first instantiation (lazy import).
# Tests may patch this name directly.
OpenAI: type | None = None


class OpenAIEmbedder:
    """Embedding provider backed by the OpenAI Embeddings API."""

    def __init__(self, settings: Settings) -> None:
        global OpenAI  # noqa: PLW0603
        if OpenAI is None:
            from openai import OpenAI as _OpenAI  # lazy import

            OpenAI = _OpenAI
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
        """Return one embedding vector per text, batching requests as needed."""
        if not texts:
            return []
        vectors: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            response = self._client.embeddings.create(input=batch, model=self._model)
            vectors.extend(item.embedding for item in response.data)
        return vectors
