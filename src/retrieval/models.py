from dataclasses import dataclass

from src.tracing.instrumentation import confidence_from_score


@dataclass(frozen=True)
class VectorStoreHit:
    chunk_id: str
    text: str
    doc_id: str
    source_path: str
    title: str
    section_heading: str | None
    chunk_index: int
    strategy: str
    similarity: float


def mean_similarity_confidence(hits: list[VectorStoreHit]) -> int | None:
    """Map the mean `similarity` across *hits* onto a 1-5 confidence score.

    Takes `list[VectorStoreHit]` (not the broader `Sequence[VectorStoreHit]`)
    specifically so it can be passed as `traced()`'s `confidence_fn` without
    widening the decorated function's inferred return type — see the note in
    `src.tracing.instrumentation.traced`.

    Returns `None` for an empty list — there's no result to be confident (or
    unconfident) about. Assumes `similarity` is already `[0, 1]`-scaled, true
    for dense/sparse/reranker hits and for `reciprocal_rank_fusion`'s
    returned hits too — RRF's own fused ranking score is used internally to
    select and order candidates but is never written onto a hit's
    `similarity`, which always stays the real pre-fusion value.
    """
    if not hits:
        return None
    return confidence_from_score(sum(hit.similarity for hit in hits) / len(hits))
