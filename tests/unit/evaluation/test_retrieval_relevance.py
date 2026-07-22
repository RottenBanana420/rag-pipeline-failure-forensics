"""Unit tests for src/evaluation/retrieval_relevance.py — pure function, no mocks."""

from __future__ import annotations

import pytest

from src.retrieval.models import VectorStoreHit


def make_hit(
    chunk_id: str = "chunk-1",
    text: str = "Some chunk text.",
    doc_id: str = "doc-1",
    source_path: str = "/repo/data/golden/corpus/01-onboarding-guide.md",
    title: str = "Onboarding Guide",
    section_heading: str | None = "Welcome & Team Structure",
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


class TestScoreRetrievalRelevance:
    def test_full_match_gives_score_one(self):
        from src.evaluation.retrieval_relevance import score_retrieval_relevance

        hits = [
            make_hit(
                source_path="/repo/data/golden/corpus/01-onboarding-guide.md",
                section_heading="Welcome & Team Structure",
            )
        ]

        result = score_retrieval_relevance(
            expected_documents=["01-onboarding-guide.md"],
            expected_sections=["Welcome & Team Structure"],
            hits=hits,
        )

        assert result.score == pytest.approx(1.0)

    def test_partial_match_gives_fractional_score(self):
        from src.evaluation.retrieval_relevance import score_retrieval_relevance

        hits = [
            make_hit(
                source_path="/repo/data/golden/corpus/01-onboarding-guide.md",
                section_heading="Local Development Setup",
            )
        ]

        result = score_retrieval_relevance(
            expected_documents=[
                "01-onboarding-guide.md",
                "02-architecture-overview.md",
            ],
            expected_sections=["Local Development Setup", "Service Ownership Table"],
            hits=hits,
        )

        assert result.score == pytest.approx(0.5)

    def test_no_match_gives_score_zero(self):
        from src.evaluation.retrieval_relevance import score_retrieval_relevance

        hits = [
            make_hit(
                source_path="/repo/data/golden/corpus/06-security-policy.md",
                section_heading="Data Classification",
            )
        ]

        result = score_retrieval_relevance(
            expected_documents=["01-onboarding-guide.md"],
            expected_sections=["Welcome & Team Structure"],
            hits=hits,
        )

        assert result.score == pytest.approx(0.0)

    def test_empty_expected_pairs_gives_none_score(self):
        """no_answer category entries have no source_documents/source_sections —
        nothing to retrieve correctly, so the score is N/A, not 0 or 1."""
        from src.evaluation.retrieval_relevance import score_retrieval_relevance

        result = score_retrieval_relevance(
            expected_documents=[],
            expected_sections=[],
            hits=[make_hit()],
        )

        assert result.score is None

    def test_empty_expected_pairs_with_no_hits_also_gives_none(self):
        from src.evaluation.retrieval_relevance import score_retrieval_relevance

        result = score_retrieval_relevance(
            expected_documents=[], expected_sections=[], hits=[]
        )

        assert result.score is None

    def test_source_path_is_matched_via_basename_not_full_path(self):
        """VectorStoreHit.source_path is str(Path) (a full path); the golden
        dataset stores bare filenames. Must compare basenames, not raw strings."""
        from src.evaluation.retrieval_relevance import score_retrieval_relevance

        hits = [
            make_hit(
                source_path="/Users/someone/Projects/repo/data/golden/corpus/01-onboarding-guide.md",
                section_heading="Welcome & Team Structure",
            )
        ]

        result = score_retrieval_relevance(
            expected_documents=["01-onboarding-guide.md"],
            expected_sections=["Welcome & Team Structure"],
            hits=hits,
        )

        assert result.score == pytest.approx(1.0)

    def test_extra_irrelevant_hits_do_not_lower_score(self):
        """Recall over expected pairs — unrelated retrieved chunks shouldn't count against it."""
        from src.evaluation.retrieval_relevance import score_retrieval_relevance

        hits = [
            make_hit(
                source_path="/repo/data/golden/corpus/01-onboarding-guide.md",
                section_heading="Welcome & Team Structure",
            ),
            make_hit(
                source_path="/repo/data/golden/corpus/06-security-policy.md",
                section_heading="Data Classification",
            ),
            make_hit(
                source_path="/repo/data/golden/corpus/07-data-retention-policy.md",
                section_heading="Retention Periods by Data Type",
            ),
        ]

        result = score_retrieval_relevance(
            expected_documents=["01-onboarding-guide.md"],
            expected_sections=["Welcome & Team Structure"],
            hits=hits,
        )

        assert result.score == pytest.approx(1.0)

    def test_duplicate_matching_hits_do_not_inflate_score(self):
        from src.evaluation.retrieval_relevance import score_retrieval_relevance

        hits = [
            make_hit(
                source_path="/repo/data/golden/corpus/01-onboarding-guide.md",
                section_heading="Welcome & Team Structure",
                chunk_id="a",
            ),
            make_hit(
                source_path="/repo/data/golden/corpus/01-onboarding-guide.md",
                section_heading="Welcome & Team Structure",
                chunk_id="b",
            ),
        ]

        result = score_retrieval_relevance(
            expected_documents=["01-onboarding-guide.md"],
            expected_sections=["Welcome & Team Structure"],
            hits=hits,
        )

        assert result.score == pytest.approx(1.0)

    def test_mismatched_array_lengths_raises(self):
        from src.evaluation.retrieval_relevance import score_retrieval_relevance

        with pytest.raises(ValueError):
            score_retrieval_relevance(
                expected_documents=["a.md", "b.md"],
                expected_sections=["Section A"],
                hits=[],
            )

    def test_multi_hop_all_pairs_matched(self):
        from src.evaluation.retrieval_relevance import score_retrieval_relevance

        hits = [
            make_hit(
                source_path="/repo/data/golden/corpus/01-onboarding-guide.md",
                section_heading="Local Development Setup",
            ),
            make_hit(
                source_path="/repo/data/golden/corpus/02-architecture-overview.md",
                section_heading="Service Ownership Table",
            ),
        ]

        result = score_retrieval_relevance(
            expected_documents=[
                "01-onboarding-guide.md",
                "02-architecture-overview.md",
            ],
            expected_sections=["Local Development Setup", "Service Ownership Table"],
            hits=hits,
        )

        assert result.score == pytest.approx(1.0)
        assert result.matched_pairs == result.expected_pairs
