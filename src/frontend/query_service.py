"""Query dashboard orchestrator — the first real caller of persist_trace.

`ask_question` wires retrieval -> generation -> citation verification ->
confidence scoring -> fallback gating together for one live request, mirroring
`diagnosis_service.py`'s "thin per-feature orchestrator" pattern: this is the
only other module in `src/frontend/` allowed to reach across the
retrieval/generation/tracing module boundaries. Every other piece
(`HybridRetriever`, `verify_citations`, `score_confidence`,
`build_fallback_response`) is documented elsewhere as a standalone,
directly-callable unit with no orchestrator wiring it in — this module is
that orchestrator, scoped to the query dashboard feature.

`build_hybrid_retriever` does the one-time wiring (embedder, vector store,
BM25 index, optional reranker) a live HybridRetriever needs; it holds no
caching logic of its own — same rule `diagnosis_service.py` follows for
`run_diagnosis` — the caller (`app_pages/query_dashboard.py`) is responsible
for wrapping it in `@st.cache_resource`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.generation.answer_generator import make_answer_generator
from src.generation.citation_verifier import (
    CitationVerificationResult,
    make_citation_judge,
    verify_citations,
)
from src.generation.confidence_scorer import (
    ConfidenceScore,
    make_completeness_judge,
    score_confidence,
)
from src.generation.fallback_response import FallbackResponse, build_fallback_response
from src.generation.prompts import build_grounded_prompt
from src.retrieval.bm25_store import BM25Store
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.embedder import make_embedder
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.models import VectorStoreHit
from src.retrieval.reranker import make_reranker
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.vector_store import make_vector_store
from src.tracing.context import collect_spans
from src.tracing.models import Trace, TraceStatus
from src.tracing.persistence import persist_trace

if TYPE_CHECKING:
    from src.config import Settings


@dataclass
class RetrieverBundle:
    """A wired HybridRetriever plus its underlying DenseRetriever.

    The dense retriever is exposed separately so the dashboard's
    hybrid-vs-dense-only comparison can reuse the same embedder/vector store
    instances instead of constructing a second one.
    """

    hybrid: HybridRetriever
    dense: DenseRetriever


def build_hybrid_retriever(settings: Settings) -> RetrieverBundle:
    """Wire an embedder, vector store, BM25 index, and optional reranker into
    a HybridRetriever. Expensive (loads an embedder model, opens the vector
    store, unpickles the BM25 index) — callers should cache the result."""
    embedder = make_embedder(settings)
    vector_store = make_vector_store(settings, embedder)
    bm25_store = BM25Store(settings)
    bm25_store.load()

    dense = DenseRetriever(embedder, vector_store)
    sparse = SparseRetriever(bm25_store, vector_store)
    reranker = make_reranker(settings) if settings.reranking_enabled else None
    hybrid = HybridRetriever(dense, sparse, settings, reranker)
    return RetrieverBundle(hybrid=hybrid, dense=dense)


@dataclass(frozen=True)
class QueryResult:
    """The full outcome of asking one question through the dashboard.

    `answer_text` is always the raw generated text, even when `fallback` is
    not `None` — the dashboard UI decides what the end user sees, but
    engineers inspecting the persisted trace can see what the model actually
    produced.
    """

    trace_id: str
    query: str
    hits: list[VectorStoreHit]
    answer_text: str
    fallback: FallbackResponse | None
    citation_results: list[CitationVerificationResult]
    confidence: ConfidenceScore
    status: TraceStatus


def ask_question(
    query: str, retriever: HybridRetriever, settings: Settings
) -> QueryResult:
    """Run the full ask pipeline for *query* and always persist a Trace.

    Runs retrieval -> generation -> citation verification -> confidence
    scoring -> fallback gating inside `collect_spans()`, so every step's
    existing `@traced`/`span()` instrumentation is captured. Persists
    "success" (a confident answer was generated) or "degraded" (the fallback
    fired) on completion; on any exception, persists "failure" with whatever
    spans completed before the error, then re-raises.
    """
    with collect_spans() as spans:
        try:
            hits = retriever.retrieve(query)

            generator = make_answer_generator(settings)
            prompt = build_grounded_prompt(query, hits)
            answer_text = generator.generate(prompt)

            citation_judge = make_citation_judge(settings)
            citation_results = verify_citations(answer_text, hits, citation_judge)

            completeness_judge = make_completeness_judge(settings)
            confidence = score_confidence(
                query,
                answer_text,
                hits,
                citation_results,
                completeness_judge,
                settings.confidence_retrieval_weight,
                settings.confidence_citation_weight,
                settings.confidence_completeness_weight,
            )

            fallback = build_fallback_response(
                hits,
                confidence.retrieval_confidence,
                settings.retrieval_confidence_threshold,
            )
        except Exception:
            persist_trace(Trace(spans=spans, status="failure"), settings)
            raise

        status: TraceStatus = "degraded" if fallback is not None else "success"
        trace = Trace(
            spans=spans,
            status=status,
            final_output=fallback.message if fallback is not None else answer_text,
            final_score=confidence.composite,
        )
        persist_trace(trace, settings)

    return QueryResult(
        trace_id=trace.trace_id,
        query=query,
        hits=hits,
        answer_text=answer_text,
        fallback=fallback,
        citation_results=citation_results,
        confidence=confidence,
        status=status,
    )
