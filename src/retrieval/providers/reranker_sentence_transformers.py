"""SentenceTransformers cross-encoder reranking provider.

``sentence_transformers`` is imported lazily inside ``__init__`` so that this
module can be imported without the package being present. Tests should patch
``sentence_transformers.CrossEncoder`` directly — Python's module cache means
the patch is visible to the inline import.
"""

from __future__ import annotations

import logging
from dataclasses import replace

from src.retrieval.models import VectorStoreHit, mean_similarity_confidence
from src.tracing.instrumentation import traced

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

logger = logging.getLogger(__name__)


class SentenceTransformersReranker:
    """Reranking provider backed by a ``sentence-transformers`` CrossEncoder."""

    def __init__(
        self, model_name: str = DEFAULT_MODEL, device: str | None = None
    ) -> None:
        import torch
        from sentence_transformers import (
            CrossEncoder,  # lazy import — not at module level
        )

        self._model_name = model_name
        # ms-marco-MiniLM-L6-v2 (and other CrossEncoders trained without an
        # activation) return raw, unbounded logits from predict() by default.
        # Forcing a sigmoid keeps `similarity` in [0, 1], matching the bounded
        # relevance scores the Cohere/Voyage rerankers already return and the
        # cosine-similarity semantics the rest of VectorStoreHit assumes —
        # score_confidence's retrieval_confidence averages this field directly.
        self._model = CrossEncoder(
            model_name, device=device, activation_fn=torch.nn.Sigmoid()
        )
        logger.info(
            "SentenceTransformersReranker loaded model=%s device=%s",
            model_name,
            self._model.device,
        )

    @property
    def provider_id(self) -> str:
        """Provider identifier including model name."""
        return f"sentence_transformers/{self._model_name}"

    @traced("ranking", confidence_fn=mean_similarity_confidence)
    def rerank(
        self, query: str, hits: list[VectorStoreHit], top_n: int
    ) -> list[VectorStoreHit]:
        """Re-score *hits* against *query* and return the top_n, sorted descending."""
        if not hits:
            return []
        pairs = [(query, hit.text) for hit in hits]
        scores = self._model.predict(pairs)
        scored = sorted(
            zip(hits, scores, strict=True), key=lambda pair: pair[1], reverse=True
        )
        return [replace(hit, similarity=float(score)) for hit, score in scored[:top_n]]
