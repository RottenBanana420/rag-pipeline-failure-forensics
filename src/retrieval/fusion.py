from dataclasses import replace

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
        result = [
            replace(hits_by_id[cid], similarity=scores[cid]) for cid in sorted_ids
        ]
        s.output = default_serialize(result)
        # Confidence comes from the *pre-fusion* similarity of the selected
        # hits, not the RRF score stamped onto `result` above — RRF scores
        # are tiny (~1/60) weighted-rank sums, not a [0,1] quality signal,
        # so feeding them through mean_similarity_confidence would always
        # bottom out at confidence 1 regardless of retrieval quality.
        s.confidence_score = mean_similarity_confidence(
            [hits_by_id[cid] for cid in sorted_ids]
        )
        return result
