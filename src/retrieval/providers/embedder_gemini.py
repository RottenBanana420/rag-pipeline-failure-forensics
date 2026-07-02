"""Google Gemini embedding provider (unified ``google-genai`` SDK).

``google.genai`` is imported lazily inside ``__init__`` so that this module
can be imported without the package being present.  Tests should patch
``google.genai.Client`` directly — Python's module cache means the patch is
visible to the inline import.

Uses the new unified ``google-genai`` package (``from google import genai``),
not the legacy ``google-generativeai`` package.
"""

from __future__ import annotations

from src.config import Settings

BATCH_SIZE = 100  # no documented fixed limit for list `contents`; conservative default

DEFAULT_MODEL = "gemini-embedding-001"

# Dimension counts for known Gemini embedding models. gemini-embedding-001
# supports configurable output_dimensionality (128-3072); default is 3072.
_MODEL_DIMENSIONS: dict[str, int] = {
    "gemini-embedding-001": 3072,
}


class GeminiEmbedder:
    """Embedding provider backed by the Google Gemini Embeddings API."""

    def __init__(self, settings: Settings) -> None:
        from google import genai  # lazy import — not at module level

        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.embedding_model

    @property
    def dimensions(self) -> int:
        """Number of dimensions produced by the configured embedding model."""
        return _MODEL_DIMENSIONS.get(self._model, 3072)

    @property
    def provider_id(self) -> str:
        """Provider identifier including model name, e.g. ``"gemini/gemini-embedding-001"``."""
        return f"gemini/{self._model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per text, batching requests as needed."""
        if not texts:
            return []
        vectors: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            # list[str] is a valid `contents` value at runtime; the stub's Union type
            # is invariant on list[str] vs. list[str | Image | File | Part], so mypy
            # rejects a plain list[str] here even though the SDK accepts it directly.
            result = self._client.models.embed_content(
                model=self._model,
                contents=batch,  # type: ignore[arg-type]
            )
            if result.embeddings is None:
                raise RuntimeError(
                    f"Gemini API returned no embeddings for model '{self._model}'"
                )
            for item in result.embeddings:
                if item.values is None:
                    raise RuntimeError(
                        f"Gemini API returned an embedding with no values for "
                        f"model '{self._model}'"
                    )
                vectors.append(item.values)
        return vectors
