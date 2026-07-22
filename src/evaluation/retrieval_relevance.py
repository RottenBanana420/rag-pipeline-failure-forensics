"""Deterministic retrieval-relevance metric — no LLM judge involved.

Recall of the golden dataset's expected (source_document, source_section)
pairs among the hits a retriever actually returned. `VectorStoreHit.source_path`
is a full path (`str(Path)`, see `src/ingestion/loader.py`), while the golden
dataset stores bare filenames — comparison must use the basename.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.retrieval.models import VectorStoreHit


@dataclass(frozen=True)
class RetrievalRelevanceResult:
    expected_pairs: set[tuple[str, str | None]]
    matched_pairs: set[tuple[str, str | None]]
    score: float | None


def score_retrieval_relevance(
    expected_documents: list[str],
    expected_sections: list[str | None],
    hits: list[VectorStoreHit],
) -> RetrievalRelevanceResult:
    """Fraction of expected (document, section) pairs found among *hits*.

    Returns `score=None` (not 0 or 1) when there are no expected pairs — a
    `no_answer` golden case has nothing to retrieve correctly, so forcing a
    score would distort aggregates rather than being excluded from them.
    """
    expected_pairs = set(zip(expected_documents, expected_sections, strict=True))
    if not expected_pairs:
        return RetrievalRelevanceResult(
            expected_pairs=set(), matched_pairs=set(), score=None
        )

    retrieved_pairs = {
        (Path(hit.source_path).name, hit.section_heading) for hit in hits
    }
    matched = expected_pairs & retrieved_pairs
    return RetrievalRelevanceResult(
        expected_pairs=expected_pairs,
        matched_pairs=matched,
        score=len(matched) / len(expected_pairs),
    )
