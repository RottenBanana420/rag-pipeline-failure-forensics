"""Fallback response for low retrieval confidence.

When `ConfidenceScore.retrieval_confidence` (from
`src.generation.confidence_scorer`) falls below a threshold, generating an
answer from weak evidence risks a confidently-worded but ungrounded response.
`build_fallback_response` is a deterministic, judge-free check on that one
signal — no LLM call is needed since the decision only needs the similarity
scores and metadata already attached to the retrieved `VectorStoreHit`s.

Like `citation_verifier.py` and `confidence_scorer.py`, this is a standalone,
directly-callable unit — the codebase has no generation orchestrator yet to
wire it in automatically. Callers compute a `ConfidenceScore` first, then
pass its `retrieval_confidence` here alongside the same `hits` used for
generation.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.retrieval.models import VectorStoreHit
from src.tracing.instrumentation import traced

FALLBACK_MESSAGE = (
    "I found some potentially related information, but not enough to "
    "answer confidently."
)


@dataclass(frozen=True)
class FallbackResponse:
    """Structured "insufficient information" response.

    Attributes:
        message: Fixed framing text (`FALLBACK_MESSAGE`), mirrors
            `INSUFFICIENT_CONTEXT_RESPONSE` in `src.generation.prompts`.
        retrieved_summary: One line per hit describing what was found —
            title (and section heading, if any) plus its similarity score.
            A fixed placeholder line if no hits were retrieved at all.
        documents_to_check: Deduplicated document identifiers worth checking
            manually, ordered by descending similarity. Hits are identified
            by `title`, unless two hits share a title with different
            `source_path`s, in which case `"title (source_path)"` is used
            instead to keep them distinguishable.
    """

    message: str
    retrieved_summary: str
    documents_to_check: list[str]


def _document_label(hit: VectorStoreHit, *, ambiguous_titles: set[str]) -> str:
    if hit.title in ambiguous_titles:
        return f"{hit.title} ({hit.source_path})"
    return hit.title


@traced("generation", is_gate=True)
def build_fallback_response(
    hits: list[VectorStoreHit],
    retrieval_confidence: float,
    threshold: float,
) -> FallbackResponse | None:
    """Return a `FallbackResponse` if `retrieval_confidence < threshold`, else `None`.

    `hits` should be the same retrieved chunks `retrieval_confidence` was
    computed from. An empty `hits` list still produces a `FallbackResponse`
    with an empty `documents_to_check` and a summary noting nothing was
    retrieved.
    """
    if retrieval_confidence >= threshold:
        return None

    if not hits:
        return FallbackResponse(
            message=FALLBACK_MESSAGE,
            retrieved_summary="No relevant documents were retrieved for this question.",
            documents_to_check=[],
        )

    ordered_hits = sorted(hits, key=lambda hit: hit.similarity, reverse=True)

    titles_seen: dict[str, set[str]] = {}
    for hit in hits:
        titles_seen.setdefault(hit.title, set()).add(hit.source_path)
    ambiguous_titles = {title for title, paths in titles_seen.items() if len(paths) > 1}

    summary_lines = []
    documents_to_check: list[str] = []
    seen_labels: set[str] = set()
    for hit in ordered_hits:
        attribution = hit.title
        if hit.section_heading:
            attribution += f" — {hit.section_heading}"
        summary_lines.append(f"{attribution} (similarity: {hit.similarity:.2f})")

        label = _document_label(hit, ambiguous_titles=ambiguous_titles)
        if label not in seen_labels:
            seen_labels.add(label)
            documents_to_check.append(label)

    return FallbackResponse(
        message=FALLBACK_MESSAGE,
        retrieved_summary="\n".join(summary_lines),
        documents_to_check=documents_to_check,
    )
