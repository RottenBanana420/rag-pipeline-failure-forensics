from src.config import Settings
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.models import VectorStoreHit
from src.retrieval.reranker import RerankerProtocol
from src.retrieval.sparse_retriever import SparseRetriever


class HybridRetriever:
    def __init__(
        self,
        dense: DenseRetriever,
        sparse: SparseRetriever,
        settings: Settings,
        reranker: RerankerProtocol | None = None,
    ) -> None:
        self._dense = dense
        self._sparse = sparse
        self._settings = settings
        self._reranker = reranker

    def retrieve(self, query: str) -> list[VectorStoreHit]:
        s = self._settings
        dense_hits = self._dense.retrieve(query, k=s.dense_top_k)
        sparse_hits = self._sparse.retrieve(query, k=s.sparse_top_k)
        candidates = reciprocal_rank_fusion(
            dense_hits,
            sparse_hits,
            dense_weight=s.dense_weight,
            sparse_weight=s.sparse_weight,
            top_n=s.rerank_candidate_pool,
        )
        if s.reranking_enabled and self._reranker is not None:
            return self._reranker.rerank(query, candidates, top_n=s.rerank_top_n)
        return candidates[: s.rerank_top_n]
