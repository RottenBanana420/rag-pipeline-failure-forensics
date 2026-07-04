"""Voyage reranking provider.

``voyageai`` is imported lazily inside ``__init__`` so that this module can be
imported without the package being present.  Tests should patch
``voyageai.Client`` directly — Python's module cache means the patch is
visible to the inline import.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from src.retrieval.models import VectorStoreHit
from src.tracing.instrumentation import traced

if TYPE_CHECKING:
    from src.config import Settings

# Voyage's recommended current-generation reranker model (the other recommended
# option being the smaller/cheaper "rerank-2.5-lite"). See
# https://docs.voyageai.com/docs/reranker
DEFAULT_MODEL = "rerank-2.5"


class VoyageReranker:
    """Reranking provider backed by the Voyage AI Rerank API."""

    def __init__(self, settings: Settings) -> None:
        import voyageai  # lazy import — not at module level

        # voyageai's __init__.py doesn't declare `Client` in __all__, so mypy's strict
        # reexport check flags this despite it being the documented public API.
        self._client = voyageai.Client(  # type: ignore[attr-defined]
            api_key=settings.voyage_api_key
        )
        self._model = settings.reranker_model

    @property
    def provider_id(self) -> str:
        """Provider identifier including model name, e.g. ``"voyage/rerank-2.5"``."""
        return f"voyage/{self._model}"

    @traced("ranking")
    def rerank(
        self, query: str, hits: list[VectorStoreHit], top_n: int
    ) -> list[VectorStoreHit]:
        """Re-score *hits* against *query* and return the top_n, sorted descending."""
        if not hits:
            return []
        result = self._client.rerank(
            query,
            [hit.text for hit in hits],
            model=self._model,
            top_k=top_n,
        )
        return [
            replace(hits[r.index], similarity=float(r.relevance_score))
            for r in result.results
        ]
