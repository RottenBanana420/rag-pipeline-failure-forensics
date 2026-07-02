"""SentenceTransformers embedding provider.

``sentence_transformers`` is imported lazily inside ``__init__`` so that this
module can be imported without the package being present.  Tests should patch
``sentence_transformers.SentenceTransformer`` directly — Python's module cache
means the patch is visible to the inline import.
"""

from __future__ import annotations

import logging

DEFAULT_MODEL = "all-MiniLM-L6-v2"

logger = logging.getLogger(__name__)


class SentenceTransformersEmbedder:
    """Embedding provider backed by the ``sentence-transformers`` library."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None) -> None:
        from sentence_transformers import (
            SentenceTransformer,  # lazy import — not at module level
        )

        self._model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        logger.info(
            "SentenceTransformersEmbedder loaded model=%s device=%s",
            model_name,
            self._model.device,
        )

    @property
    def dimensions(self) -> int:
        """Number of dimensions produced by the loaded model."""
        dim = self._model.get_embedding_dimension()
        if dim is None:
            raise RuntimeError(
                f"Could not determine embedding dimension for model '{self._model_name}'"
            )
        result: int = dim  # mypy stubs type dim as Any; annotation narrows without a cast
        return result

    @property
    def provider_id(self) -> str:
        """Provider identifier including model name."""
        return f"sentence_transformers/{self._model_name}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per text as a list of floats."""
        if not texts:
            return []
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return [vec.tolist() for vec in embeddings]
