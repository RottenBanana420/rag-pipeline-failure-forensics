"""Cohere reranking provider.

``cohere`` is imported lazily inside ``__init__`` so that this module can be
imported without the package being present.  Tests should patch
``cohere.ClientV2`` directly — Python's module cache means the patch is
visible to the inline import.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from src.retrieval.models import VectorStoreHit, mean_similarity_confidence
from src.tracing.instrumentation import traced

if TYPE_CHECKING:
    from src.config import Settings

# Cohere's newest and most performant foundational reranking model, superseding
# rerank-v3.5 (released December 2025). See https://docs.cohere.com/changelog/rerank-v4.0
DEFAULT_MODEL = "rerank-v4.0-pro"


class CohereReranker:
    """Reranking provider backed by the Cohere Rerank API."""

    def __init__(self, settings: Settings) -> None:
        import cohere  # lazy import — not at module level

        self._client = cohere.ClientV2(api_key=settings.cohere_api_key)
        self._model = settings.reranker_model

    @property
    def provider_id(self) -> str:
        """Provider identifier including model name, e.g. ``"cohere/rerank-v4.0-pro"``."""
        return f"cohere/{self._model}"

    @traced("ranking", confidence_fn=mean_similarity_confidence)
    def rerank(
        self, query: str, hits: list[VectorStoreHit], top_n: int
    ) -> list[VectorStoreHit]:
        """Re-score *hits* against *query* and return the top_n, sorted descending."""
        if not hits:
            return []
        response = self._client.v2.rerank(
            model=self._model,
            query=query,
            documents=[hit.text for hit in hits],
            top_n=top_n,
        )
        return [
            replace(hits[result.index], similarity=float(result.relevance_score))
            for result in response.results
        ]
