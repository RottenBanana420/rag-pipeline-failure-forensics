"""Reranker module — protocol definition and factory.

``RerankerProtocol`` defines the structural interface for cross-encoder
reranking providers. ``make_reranker`` is a factory that reads
``settings.reranker_provider`` and returns the appropriate provider instance
with lazy imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from src.config import Settings
    from src.retrieval.models import VectorStoreHit


@runtime_checkable
class RerankerProtocol(Protocol):
    """Structural interface that every reranking provider must satisfy."""

    def rerank(
        self, query: str, hits: list[VectorStoreHit], top_n: int
    ) -> list[VectorStoreHit]:
        """Re-score *hits* against *query* and return the top_n, sorted descending."""
        ...

    @property
    def provider_id(self) -> str:
        """Short identifier for the provider, e.g. ``"sentence_transformers/cross-encoder/..."``."""
        ...


def make_reranker(settings: Settings) -> RerankerProtocol:
    """Return a reranker instance for the provider specified in *settings*.

    Provider modules are imported lazily inside this function so that importing
    ``src.retrieval.reranker`` does not pull in optional heavy dependencies
    unless they are actually needed.

    Raises:
        ValueError: If ``settings.reranker_provider`` is not a recognised value.
    """
    provider = settings.reranker_provider

    if provider == "sentence_transformers":
        from src.retrieval.providers.reranker_sentence_transformers import (
            SentenceTransformersReranker as _STReranker,
        )

        device = None if settings.reranker_device == "auto" else settings.reranker_device
        return _STReranker(model_name=settings.reranker_model, device=device)

    # sentence_transformers' global default — used below to detect whether the
    # user has customized reranker_model at all. Cohere's and Voyage's model
    # names both start with "rerank-", so (unlike make_embedder) a prefix check
    # cannot distinguish "the user meant this provider's default" from "the user
    # meant the other provider's default" — an equality check against the
    # sentence_transformers default is the only unambiguous signal available.
    _ST_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

    if provider == "cohere":
        from src.retrieval.providers.reranker_cohere import (
            DEFAULT_MODEL as _COHERE_DEFAULT_MODEL,
        )
        from src.retrieval.providers.reranker_cohere import (
            CohereReranker as _CohereReranker,
        )

        model_name = (
            _COHERE_DEFAULT_MODEL
            if settings.reranker_model == _ST_DEFAULT_MODEL
            else settings.reranker_model
        )
        return _CohereReranker(
            settings.model_copy(update={"reranker_model": model_name})
        )

    if provider == "voyage":
        from src.retrieval.providers.reranker_voyage import (
            DEFAULT_MODEL as _VOYAGE_DEFAULT_MODEL,
        )
        from src.retrieval.providers.reranker_voyage import (
            VoyageReranker as _VoyageReranker,
        )

        model_name = (
            _VOYAGE_DEFAULT_MODEL
            if settings.reranker_model == _ST_DEFAULT_MODEL
            else settings.reranker_model
        )
        return _VoyageReranker(
            settings.model_copy(update={"reranker_model": model_name})
        )

    valid = "sentence_transformers, cohere, voyage"
    raise ValueError(
        f"Unknown reranker provider: {provider!r}. Valid providers are: {valid}"
    )
