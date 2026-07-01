"""SentenceTransformers embedding provider.

``sentence_transformers`` is imported lazily inside the constructor so that
this module can be imported without the package being present.  Tests should
patch ``src.retrieval.providers.embedder_sentence_transformers.SentenceTransformer``
— the name is bound once on the first instantiation and cached as a module-level
name so subsequent patches also work.
"""

from __future__ import annotations

DEFAULT_MODEL = "all-MiniLM-L6-v2"

# Module-level placeholder — populated on first instantiation (lazy import).
# Tests may patch this name directly.
SentenceTransformer: type | None = None


class SentenceTransformersEmbedder:
    """Embedding provider backed by the ``sentence-transformers`` library."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        global SentenceTransformer  # noqa: PLW0603
        if SentenceTransformer is None:
            from sentence_transformers import (  # lazy import
                SentenceTransformer as _SentenceTransformer,
            )

            SentenceTransformer = _SentenceTransformer
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)

    @property
    def dimensions(self) -> int:
        """Number of dimensions produced by the loaded model."""
        dim = self._model.get_embedding_dimension()
        if dim is None:
            raise RuntimeError(
                f"Could not determine embedding dimension for model '{self._model_name}'"
            )
        return int(dim)

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
