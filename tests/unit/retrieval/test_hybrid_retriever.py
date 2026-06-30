from unittest.mock import MagicMock

from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.models import VectorStoreHit
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
