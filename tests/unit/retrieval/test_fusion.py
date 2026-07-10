from dataclasses import replace

import pytest

from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.models import VectorStoreHit
from src.tracing.context import collect_spans


def _hit(chunk_id: str, similarity: float = 0.9) -> VectorStoreHit:
    return VectorStoreHit(
        chunk_id=chunk_id,
        text="text",
        doc_id="doc1",
        source_path="/p",
        title="T",
        section_heading=None,
        chunk_index=0,
        strategy="fixed_size",
        similarity=similarity,
    )


class TestRRFEmptyInputs:
    def test_both_empty_returns_empty(self):
        assert reciprocal_rank_fusion([], [], top_n=5) == []

    def test_dense_only_returns_top_n(self):
        hits = [_hit(f"c{i}") for i in range(10)]
        result = reciprocal_rank_fusion(
            hits, [], dense_weight=1.0, sparse_weight=0.0, top_n=3
        )
        assert len(result) == 3

    def test_sparse_only_returns_top_n(self):
        hits = [_hit(f"c{i}") for i in range(10)]
        result = reciprocal_rank_fusion(
            [], hits, dense_weight=0.0, sparse_weight=1.0, top_n=3
        )
        assert len(result) == 3


class TestRRFScoring:
    def test_rank1_beats_rank2_same_weight(self):
        d = [_hit("rank1"), _hit("rank2")]
        result = reciprocal_rank_fusion(
            d, [], dense_weight=1.0, sparse_weight=0.0, top_n=2
        )
        ids = [h.chunk_id for h in result]
        assert ids == ["rank1", "rank2"]

    def test_overlap_chunk_scores_higher_than_single_list(self):
        dense = [_hit("shared"), _hit("dense_only")]
        sparse = [_hit("shared"), _hit("sparse_only")]
        result = reciprocal_rank_fusion(dense, sparse, top_n=3)
        assert result[0].chunk_id == "shared"

    def test_no_overlap_merged_and_sorted_by_rrf_rank(self):
        # dense_weight=0.7/sparse_weight=0.3 defaults: d1 (0.7/61) > d2 (0.7/62)
        # > s1 (0.3/61) > s2 (0.3/62), even though every hit's own similarity
        # is the same 0.9 default — ordering is RRF-rank-based, not
        # similarity-based, since similarity no longer carries the RRF score.
        dense = [_hit("d1"), _hit("d2")]
        sparse = [_hit("s1"), _hit("s2")]
        result = reciprocal_rank_fusion(dense, sparse, top_n=4)
        assert [h.chunk_id for h in result] == ["d1", "d2", "s1", "s2"]

    def test_similarity_field_preserves_pre_fusion_similarity(self):
        hits = [_hit("c1", similarity=0.99)]
        result = reciprocal_rank_fusion(
            hits, [], dense_weight=1.0, sparse_weight=0.0, top_n=1
        )
        assert result[0].similarity == pytest.approx(0.99, rel=1e-6)

    def test_similarity_field_never_carries_rrf_score(self):
        hits = [_hit("c1", similarity=0.99)]
        result = reciprocal_rank_fusion(
            hits, [], dense_weight=1.0, sparse_weight=0.0, top_n=1
        )
        rrf_score = 1.0 / (60 + 1)
        assert result[0].similarity != pytest.approx(rrf_score, rel=1e-6)

    def test_similarity_bounded_zero_to_one(self):
        dense = [_hit(f"d{i}", similarity=0.1 * (i + 1)) for i in range(5)]
        sparse = [_hit(f"s{i}", similarity=0.1 * (i + 1)) for i in range(5)]
        result = reciprocal_rank_fusion(dense, sparse, top_n=10)
        assert all(0.0 <= h.similarity <= 1.0 for h in result)

    def test_top_n_limits_output(self):
        dense = [_hit(f"d{i}") for i in range(20)]
        result = reciprocal_rank_fusion(dense, [], top_n=5)
        assert len(result) == 5

    def test_top_n_larger_than_results_returns_all(self):
        result = reciprocal_rank_fusion([_hit("only")], [], top_n=10)
        assert len(result) == 1

    def test_dense_hit_metadata_used_when_chunk_in_both_lists(self):
        dense_hit = _hit("shared")
        sparse_hit = replace(_hit("shared"), title="SparseTitle")
        result = reciprocal_rank_fusion([dense_hit], [sparse_hit], top_n=1)
        assert result[0].title == "T"

    def test_zero_weight_list_excluded_from_top_n(self):
        dense = [_hit("d1"), _hit("d2")]
        sparse = [_hit("s1")]
        result = reciprocal_rank_fusion(
            dense, sparse, dense_weight=1.0, sparse_weight=0.0, top_n=2
        )
        chunk_ids = {h.chunk_id for h in result}
        assert chunk_ids == {"d1", "d2"}


class TestFusionTracing:
    def test_records_retrieval_span(self):
        hits = [_hit("c1")]

        with collect_spans() as spans:
            reciprocal_rank_fusion(hits, [], top_n=1)

        assert len(spans) == 1
        assert spans[0].step == "retrieval"
        assert spans[0].error is None

    def test_noop_outside_collect_spans(self):
        reciprocal_rank_fusion([], [], top_n=5)

    def test_confidence_score_uses_underlying_similarity_not_rrf_score(self):
        # The RRF score itself (~1/61 here) would map to confidence 1 if used
        # directly — confidence must come from the pre-fusion similarity
        # (1.0) instead, which maps to 5.
        hits = [_hit("c1", similarity=1.0)]

        with collect_spans() as spans:
            reciprocal_rank_fusion(
                hits, [], dense_weight=1.0, sparse_weight=0.0, top_n=1
            )

        assert spans[0].confidence_score == 5

    def test_confidence_score_averages_similarity_of_selected_hits(self):
        dense = [_hit("d1", similarity=1.0)]
        sparse = [_hit("s1", similarity=0.0)]

        with collect_spans() as spans:
            reciprocal_rank_fusion(dense, sparse, top_n=2)

        assert spans[0].confidence_score == 3

    def test_no_hits_leaves_confidence_score_none(self):
        with collect_spans() as spans:
            reciprocal_rank_fusion([], [], top_n=5)

        assert spans[0].confidence_score is None
