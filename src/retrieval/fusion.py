from src.retrieval.models import VectorStoreHit, mean_similarity_confidence
from src.tracing.instrumentation import default_serialize, span

_RRF_K = 60


def reciprocal_rank_fusion(
    dense_hits: list[VectorStoreHit],
    sparse_hits: list[VectorStoreHit],
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    top_n: int = 5,
) -> list[VectorStoreHit]:
    arguments = {
        "dense_hits": dense_hits,
        "sparse_hits": sparse_hits,
        "dense_weight": dense_weight,
        "sparse_weight": sparse_weight,
        "top_n": top_n,
    }
    with span("retrieval", input=default_serialize(arguments)) as s:
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
        # The RRF fused score (`scores`, a ~1/60 weighted-rank sum) selects
        # and orders the top_n candidates above, but is never written onto a
        # hit's `similarity` — it isn't a [0,1] quality signal comparable
        # across retrievers (that's RRF's whole premise: fuse by rank, not
        # score). Each returned hit keeps its own pre-fusion similarity
        # (dense cosine, or sparse max-normalized BM25) so downstream
        # consumers that assume [0,1]-scaled `similarity` get a real signal
        # whether or not a reranker runs afterward.
        result = [hits_by_id[cid] for cid in sorted_ids]
        s.output = default_serialize(result)
        s.confidence_score = mean_similarity_confidence(result)
        return result
