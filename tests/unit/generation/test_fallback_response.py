"""Unit tests for the low-retrieval-confidence fallback response."""

import dataclasses

import pytest

from src.generation.fallback_response import (
    FALLBACK_MESSAGE,
    FallbackResponse,
    build_fallback_response,
)
from src.retrieval.models import VectorStoreHit


def make_hit(
    chunk_id: str = "chunk-1",
    text: str = "Paris is the capital of France.",
    doc_id: str = "doc-1",
    source_path: str = "/docs/geography.md",
    title: str = "Geography Facts",
    section_heading: str | None = "Capitals",
    chunk_index: int = 0,
    strategy: str = "fixed_size",
    similarity: float = 0.9,
) -> VectorStoreHit:
    return VectorStoreHit(
        chunk_id=chunk_id,
        text=text,
        doc_id=doc_id,
        source_path=source_path,
        title=title,
        section_heading=section_heading,
        chunk_index=chunk_index,
        strategy=strategy,
        similarity=similarity,
    )


class TestBuildFallbackResponse:
    def test_returns_none_when_confidence_above_threshold(self):
        hits = [make_hit(similarity=0.9)]

        result = build_fallback_response(hits, retrieval_confidence=0.9, threshold=0.5)

        assert result is None

    def test_returns_none_when_confidence_equal_to_threshold(self):
        hits = [make_hit(similarity=0.5)]

        result = build_fallback_response(hits, retrieval_confidence=0.5, threshold=0.5)

        assert result is None

    def test_returns_fallback_when_confidence_below_threshold(self):
        hits = [make_hit(similarity=0.2)]

        result = build_fallback_response(hits, retrieval_confidence=0.2, threshold=0.5)

        assert result is not None
        assert result.message == FALLBACK_MESSAGE

    def test_empty_hits_triggers_fallback_with_empty_documents(self):
        result = build_fallback_response([], retrieval_confidence=0.0, threshold=0.5)

        assert result is not None
        assert result.documents_to_check == []
        assert "No relevant documents" in result.retrieved_summary

    def test_summary_lists_every_hit(self):
        hits = [
            make_hit(
                chunk_id="c1", title="Doc A", section_heading="Intro", similarity=0.3
            ),
            make_hit(
                chunk_id="c2", title="Doc B", section_heading=None, similarity=0.1
            ),
        ]

        result = build_fallback_response(hits, retrieval_confidence=0.2, threshold=0.5)

        assert result is not None
        assert "Doc A — Intro" in result.retrieved_summary
        assert "Doc B" in result.retrieved_summary

    def test_documents_to_check_ordered_by_similarity_descending(self):
        hits = [
            make_hit(chunk_id="c1", title="Low Match", similarity=0.1),
            make_hit(chunk_id="c2", title="Higher Match", similarity=0.4),
        ]

        result = build_fallback_response(hits, retrieval_confidence=0.25, threshold=0.5)

        assert result is not None
        assert result.documents_to_check == ["Higher Match", "Low Match"]

    def test_duplicate_titles_deduplicated(self):
        hits = [
            make_hit(
                chunk_id="c1",
                title="Doc A",
                source_path="/docs/a.md",
                similarity=0.3,
            ),
            make_hit(
                chunk_id="c2",
                title="Doc A",
                source_path="/docs/a.md",
                similarity=0.2,
            ),
        ]

        result = build_fallback_response(hits, retrieval_confidence=0.25, threshold=0.5)

        assert result is not None
        assert result.documents_to_check == ["Doc A"]

    def test_same_title_different_source_disambiguated(self):
        hits = [
            make_hit(
                chunk_id="c1",
                title="Doc A",
                source_path="/docs/a1.md",
                similarity=0.3,
            ),
            make_hit(
                chunk_id="c2",
                title="Doc A",
                source_path="/docs/a2.md",
                similarity=0.2,
            ),
        ]

        result = build_fallback_response(hits, retrieval_confidence=0.25, threshold=0.5)

        assert result is not None
        assert result.documents_to_check == [
            "Doc A (/docs/a1.md)",
            "Doc A (/docs/a2.md)",
        ]

    def test_records_generation_span(self):
        from src.tracing.context import collect_spans

        hits = [make_hit(similarity=0.2)]

        with collect_spans() as spans:
            build_fallback_response(hits, retrieval_confidence=0.2, threshold=0.5)

        assert len(spans) == 1
        assert spans[0].step == "generation"
        assert spans[0].error is None

    def test_records_span_even_when_returning_none(self):
        from src.tracing.context import collect_spans

        hits = [make_hit(similarity=0.9)]

        with collect_spans() as spans:
            result = build_fallback_response(
                hits, retrieval_confidence=0.9, threshold=0.5
            )

        assert result is None
        assert len(spans) == 1

    def test_noop_outside_collect_spans(self):
        build_fallback_response([make_hit()], retrieval_confidence=0.9, threshold=0.5)


class TestFallbackResponse:
    def test_is_frozen(self):
        response = FallbackResponse(
            message="msg", retrieved_summary="summary", documents_to_check=[]
        )

        with pytest.raises(dataclasses.FrozenInstanceError):
            response.message = "changed"  # type: ignore[misc]
