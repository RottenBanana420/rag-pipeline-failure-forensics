from unittest.mock import MagicMock

import pytest

from src.retrieval.bm25_store import BM25Store
from src.retrieval.models import VectorStoreHit
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.vector_store import VectorStore
from src.tracing.context import collect_spans


def _hit(**kwargs) -> VectorStoreHit:
    defaults = dict(
        chunk_id="c-000",
        text="sample",
        doc_id="d-000",
        source_path="/doc.md",
        title="Doc",
        section_heading=None,
        chunk_index=0,
        strategy="fixed_size",
        similarity=0.0,
    )
    return VectorStoreHit(**{**defaults, **kwargs})


class TestSparseRetriever:
    def test_retrieve_calls_get_scores_with_query(self):
        bm25 = MagicMock(spec=BM25Store)
        bm25.get_scores.return_value = []
        vs = MagicMock(spec=VectorStore)

        SparseRetriever(bm25, vs).retrieve("error code 404")

        bm25.get_scores.assert_called_once_with("error code 404")

    def test_retrieve_empty_store_returns_empty(self):
        bm25 = MagicMock(spec=BM25Store)
        bm25.get_scores.return_value = []
        vs = MagicMock(spec=VectorStore)

        result = SparseRetriever(bm25, vs).retrieve("q")

        assert result == []
        vs.get_by_ids.assert_not_called()

    def test_retrieve_all_zero_scores_returns_empty(self):
        bm25 = MagicMock(spec=BM25Store)
        bm25.get_scores.return_value = [("c-000", 0.0), ("c-001", 0.0)]
        vs = MagicMock(spec=VectorStore)

        result = SparseRetriever(bm25, vs).retrieve("q")

        assert result == []
        vs.get_by_ids.assert_not_called()

    def test_retrieve_hydrates_via_get_by_ids(self):
        bm25 = MagicMock(spec=BM25Store)
        bm25.get_scores.return_value = [("c-000", 3.5), ("c-001", 1.2)]
        vs = MagicMock(spec=VectorStore)
        vs.get_by_ids.return_value = [_hit(chunk_id="c-000"), _hit(chunk_id="c-001")]

        SparseRetriever(bm25, vs).retrieve("q")

        vs.get_by_ids.assert_called_once_with(["c-000", "c-001"])

    def test_retrieve_replaces_similarity_with_normalized_bm25_score(self):
        bm25 = MagicMock(spec=BM25Store)
        bm25.get_scores.return_value = [("c-000", 4.2), ("c-001", 1.8)]
        vs = MagicMock(spec=VectorStore)
        vs.get_by_ids.return_value = [
            _hit(chunk_id="c-000", similarity=0.0),
            _hit(chunk_id="c-001", similarity=0.0),
        ]

        hits = SparseRetriever(bm25, vs).retrieve("q")

        score_map = {h.chunk_id: h.similarity for h in hits}
        assert score_map["c-000"] == pytest.approx(1.0)
        assert score_map["c-001"] == pytest.approx(1.8 / 4.2)

    def test_retrieve_scores_normalized_to_0_1(self):
        bm25 = MagicMock(spec=BM25Store)
        bm25.get_scores.return_value = [("c-000", 8.4), ("c-001", 4.2), ("c-002", 2.1)]
        vs = MagicMock(spec=VectorStore)
        vs.get_by_ids.return_value = [
            _hit(chunk_id="c-000"),
            _hit(chunk_id="c-001"),
            _hit(chunk_id="c-002"),
        ]

        hits = SparseRetriever(bm25, vs).retrieve("q")

        assert all(0.0 <= h.similarity <= 1.0 for h in hits)

    def test_retrieve_top_result_has_similarity_1(self):
        bm25 = MagicMock(spec=BM25Store)
        bm25.get_scores.return_value = [("c-000", 5.5), ("c-001", 2.2)]
        vs = MagicMock(spec=VectorStore)
        vs.get_by_ids.return_value = [_hit(chunk_id="c-000"), _hit(chunk_id="c-001")]

        hits = SparseRetriever(bm25, vs).retrieve("q")

        assert hits[0].similarity == pytest.approx(1.0)

    def test_retrieve_returns_hits_in_bm25_rank_order(self):
        bm25 = MagicMock(spec=BM25Store)
        bm25.get_scores.return_value = [("c-001", 1.0), ("c-000", 5.0), ("c-002", 2.5)]
        vs = MagicMock(spec=VectorStore)
        vs.get_by_ids.return_value = [
            _hit(chunk_id="c-000"),
            _hit(chunk_id="c-001"),
            _hit(chunk_id="c-002"),
        ]

        hits = SparseRetriever(bm25, vs).retrieve("q")

        assert [h.chunk_id for h in hits] == ["c-000", "c-002", "c-001"]

    def test_retrieve_limits_to_k(self):
        bm25 = MagicMock(spec=BM25Store)
        bm25.get_scores.return_value = [
            (f"c-{i:03d}", float(10 - i)) for i in range(10)
        ]
        vs = MagicMock(spec=VectorStore)
        vs.get_by_ids.return_value = [_hit(chunk_id=f"c-{i:03d}") for i in range(3)]

        SparseRetriever(bm25, vs).retrieve("q", k=3)

        called_ids = vs.get_by_ids.call_args.args[0]
        assert len(called_ids) == 3

    def test_retrieve_default_k_is_10(self):
        bm25 = MagicMock(spec=BM25Store)
        bm25.get_scores.return_value = [
            (f"c-{i:03d}", float(20 - i)) for i in range(20)
        ]
        vs = MagicMock(spec=VectorStore)
        vs.get_by_ids.return_value = [_hit(chunk_id=f"c-{i:03d}") for i in range(10)]

        SparseRetriever(bm25, vs).retrieve("q")

        called_ids = vs.get_by_ids.call_args.args[0]
        assert len(called_ids) == 10

    def test_retrieve_filters_zero_score_before_hydration(self):
        bm25 = MagicMock(spec=BM25Store)
        bm25.get_scores.return_value = [("c-000", 3.0), ("c-001", 0.0), ("c-002", 1.5)]
        vs = MagicMock(spec=VectorStore)
        vs.get_by_ids.return_value = [
            _hit(chunk_id="c-000"),
            _hit(chunk_id="c-002"),
        ]

        SparseRetriever(bm25, vs).retrieve("q")

        called_ids = vs.get_by_ids.call_args.args[0]
        assert "c-001" not in called_ids


class TestSparseRetrieverTracing:
    def test_retrieve_records_retrieval_span(self):
        bm25 = MagicMock(spec=BM25Store)
        bm25.get_scores.return_value = [("c-000", 3.5)]
        vs = MagicMock(spec=VectorStore)
        vs.get_by_ids.return_value = [_hit(chunk_id="c-000")]

        with collect_spans() as spans:
            SparseRetriever(bm25, vs).retrieve("q")

        assert len(spans) == 1
        assert spans[0].step == "retrieval"
        assert spans[0].error is None

    def test_retrieve_noop_outside_collect_spans(self):
        bm25 = MagicMock(spec=BM25Store)
        bm25.get_scores.return_value = []
        vs = MagicMock(spec=VectorStore)

        SparseRetriever(bm25, vs).retrieve("q")

    def test_retrieve_sets_confidence_score_from_mean_similarity(self):
        bm25 = MagicMock(spec=BM25Store)
        bm25.get_scores.return_value = [("c-000", 3.5)]
        vs = MagicMock(spec=VectorStore)
        vs.get_by_ids.return_value = [_hit(chunk_id="c-000")]

        with collect_spans() as spans:
            SparseRetriever(bm25, vs).retrieve("q")

        # Single hit normalizes to similarity 1.0 -> confidence 5.
        assert spans[0].confidence_score == 5

    def test_retrieve_no_hits_leaves_confidence_score_none(self):
        bm25 = MagicMock(spec=BM25Store)
        bm25.get_scores.return_value = []
        vs = MagicMock(spec=VectorStore)

        with collect_spans() as spans:
            SparseRetriever(bm25, vs).retrieve("q")

        assert spans[0].confidence_score is None
