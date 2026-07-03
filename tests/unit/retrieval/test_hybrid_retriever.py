from unittest.mock import MagicMock

from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.models import VectorStoreHit
from src.retrieval.reranker import RerankerProtocol
from src.retrieval.sparse_retriever import SparseRetriever


def _hit(chunk_id: str, similarity: float = 0.5) -> VectorStoreHit:
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


class TestHybridRetrieverRetrieve:
    def test_calls_dense_with_configured_k(self, settings):
        dense = MagicMock(spec=DenseRetriever)
        dense.retrieve.return_value = [_hit("d1")]
        sparse = MagicMock(spec=SparseRetriever)
        sparse.retrieve.return_value = []
        HybridRetriever(dense, sparse, settings).retrieve("q")
        dense.retrieve.assert_called_once_with("q", k=settings.dense_top_k)

    def test_calls_sparse_with_configured_k(self, settings):
        dense = MagicMock(spec=DenseRetriever)
        dense.retrieve.return_value = []
        sparse = MagicMock(spec=SparseRetriever)
        sparse.retrieve.return_value = [_hit("s1")]
        HybridRetriever(dense, sparse, settings).retrieve("q")
        sparse.retrieve.assert_called_once_with("q", k=settings.sparse_top_k)

    def test_returns_list_of_vector_store_hits(self, settings):
        dense = MagicMock(spec=DenseRetriever)
        dense.retrieve.return_value = [_hit("d1"), _hit("d2")]
        sparse = MagicMock(spec=SparseRetriever)
        sparse.retrieve.return_value = [_hit("s1")]
        result = HybridRetriever(dense, sparse, settings).retrieve("q")
        assert isinstance(result, list)
        assert all(isinstance(h, VectorStoreHit) for h in result)

    def test_result_limited_to_rerank_top_n(self, settings):
        dense = MagicMock(spec=DenseRetriever)
        dense.retrieve.return_value = [_hit(f"d{i}") for i in range(10)]
        sparse = MagicMock(spec=SparseRetriever)
        sparse.retrieve.return_value = [_hit(f"s{i}") for i in range(10)]
        result = HybridRetriever(dense, sparse, settings).retrieve("q")
        assert len(result) <= settings.rerank_top_n

    def test_overlap_chunk_ranks_first(self, settings):
        shared = _hit("shared")
        dense = MagicMock(spec=DenseRetriever)
        dense.retrieve.return_value = [shared, _hit("d_only")]
        sparse = MagicMock(spec=SparseRetriever)
        sparse.retrieve.return_value = [shared, _hit("s_only")]
        result = HybridRetriever(dense, sparse, settings).retrieve("q")
        assert result[0].chunk_id == "shared"

    def test_empty_dense_returns_sparse_results(self, settings):
        dense = MagicMock(spec=DenseRetriever)
        dense.retrieve.return_value = []
        sparse = MagicMock(spec=SparseRetriever)
        sparse.retrieve.return_value = [_hit("s1"), _hit("s2")]
        result = HybridRetriever(dense, sparse, settings).retrieve("q")
        assert len(result) > 0

    def test_empty_sparse_returns_dense_results(self, settings):
        dense = MagicMock(spec=DenseRetriever)
        dense.retrieve.return_value = [_hit("d1"), _hit("d2")]
        sparse = MagicMock(spec=SparseRetriever)
        sparse.retrieve.return_value = []
        result = HybridRetriever(dense, sparse, settings).retrieve("q")
        assert len(result) > 0

    def test_both_empty_returns_empty(self, settings):
        dense = MagicMock(spec=DenseRetriever)
        dense.retrieve.return_value = []
        sparse = MagicMock(spec=SparseRetriever)
        sparse.retrieve.return_value = []
        result = HybridRetriever(dense, sparse, settings).retrieve("q")
        assert result == []


class TestHybridRetrieverReranking:
    def test_falls_back_to_rrf_slice_when_reranker_not_provided(self, settings):
        dense = MagicMock(spec=DenseRetriever)
        dense.retrieve.return_value = [_hit(f"d{i}") for i in range(10)]
        sparse = MagicMock(spec=SparseRetriever)
        sparse.retrieve.return_value = [_hit(f"s{i}") for i in range(10)]
        result = HybridRetriever(dense, sparse, settings).retrieve("q")
        assert len(result) == settings.rerank_top_n

    def test_falls_back_to_rrf_slice_when_reranking_disabled(self, settings, monkeypatch):
        monkeypatch.setattr(settings, "reranking_enabled", False)
        dense = MagicMock(spec=DenseRetriever)
        dense.retrieve.return_value = [_hit(f"d{i}") for i in range(10)]
        sparse = MagicMock(spec=SparseRetriever)
        sparse.retrieve.return_value = [_hit(f"s{i}") for i in range(10)]
        reranker = MagicMock(spec=RerankerProtocol)

        result = HybridRetriever(dense, sparse, settings, reranker=reranker).retrieve("q")

        reranker.rerank.assert_not_called()
        assert len(result) == settings.rerank_top_n

    def test_delegates_to_reranker_when_enabled(self, settings):
        dense = MagicMock(spec=DenseRetriever)
        dense.retrieve.return_value = [_hit(f"d{i}") for i in range(10)]
        sparse = MagicMock(spec=SparseRetriever)
        sparse.retrieve.return_value = [_hit(f"s{i}") for i in range(10)]
        reranker = MagicMock(spec=RerankerProtocol)
        reranked = [_hit("reranked1"), _hit("reranked2")]
        reranker.rerank.return_value = reranked

        result = HybridRetriever(dense, sparse, settings, reranker=reranker).retrieve("q")

        reranker.rerank.assert_called_once()
        call_args = reranker.rerank.call_args
        assert call_args.args[0] == "q"
        assert len(call_args.args[1]) == settings.rerank_candidate_pool
        assert call_args.kwargs == {"top_n": settings.rerank_top_n}
        assert result == reranked

    def test_rrf_called_with_candidate_pool_not_final_top_n(self, settings):
        dense = MagicMock(spec=DenseRetriever)
        dense.retrieve.return_value = [_hit(f"d{i}") for i in range(10)]
        sparse = MagicMock(spec=SparseRetriever)
        sparse.retrieve.return_value = [_hit(f"s{i}") for i in range(10)]
        reranker = MagicMock(spec=RerankerProtocol)
        reranker.rerank.side_effect = lambda query, hits, top_n: hits[:top_n]

        assert settings.rerank_candidate_pool > settings.rerank_top_n
        HybridRetriever(dense, sparse, settings, reranker=reranker).retrieve("q")

        candidates_passed_to_reranker = reranker.rerank.call_args.args[1]
        assert len(candidates_passed_to_reranker) == settings.rerank_candidate_pool
