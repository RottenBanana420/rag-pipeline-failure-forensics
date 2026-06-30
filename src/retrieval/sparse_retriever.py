import dataclasses

from src.retrieval.bm25_store import BM25Store
from src.retrieval.models import VectorStoreHit
from src.retrieval.vector_store import VectorStore

_DEFAULT_K = 10


class SparseRetriever:
    def __init__(self, bm25_store: BM25Store, vector_store: VectorStore) -> None:
        self._bm25_store = bm25_store
        self._vector_store = vector_store

    def retrieve(self, query: str, k: int = _DEFAULT_K) -> list[VectorStoreHit]:
        scores = self._bm25_store.get_scores(query)
        if not scores:
            return []
        top_k = sorted(scores, key=lambda x: x[1], reverse=True)[:k]
        top_k = [(cid, s) for cid, s in top_k if s > 0.0]
        if not top_k:
            return []
        max_score = top_k[0][1]
        top_k = [(cid, s / max_score) for cid, s in top_k]
        ids = [cid for cid, _ in top_k]
        score_map = {cid: s for cid, s in top_k}
        hits = self._vector_store.get_by_ids(ids)
        hit_map = {h.chunk_id: h for h in hits}
        return [
            dataclasses.replace(hit_map[cid], similarity=score_map[cid])
            for cid in ids
            if cid in hit_map
        ]
