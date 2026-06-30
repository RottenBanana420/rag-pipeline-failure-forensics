from dataclasses import replace

from src.retrieval.models import VectorStoreHit

_RRF_K = 60


def reciprocal_rank_fusion(
    dense_hits: list[VectorStoreHit],
    sparse_hits: list[VectorStoreHit],
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    top_n: int = 5,
) -> list[VectorStoreHit]:
    scores: dict[str, float] = {}
    hits_by_id: dict[str, VectorStoreHit] = {}

    for rank, hit in enumerate(dense_hits, start=1):
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + dense_weight / (
            _RRF_K + rank
        )
        hits_by_id[hit.chunk_id] = hit

    for rank, hit in enumerate(sparse_hits, start=1):
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + sparse_weight / (
            _RRF_K + rank
        )
        hits_by_id.setdefault(hit.chunk_id, hit)

    sorted_ids = sorted(scores, key=scores.__getitem__, reverse=True)[:top_n]
    return [replace(hits_by_id[cid], similarity=scores[cid]) for cid in sorted_ids]
