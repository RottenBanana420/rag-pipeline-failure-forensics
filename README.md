# RAG Pipeline with Integrated Failure Forensics

Production-grade Retrieval-Augmented Generation system with built-in observability.
Every pipeline step is traced; failures are diagnosed automatically via backward span analysis.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # embeddings/retrieval default to local sentence_transformers (no key needed),
                        # but ANTHROPIC_API_KEY is required by default too — answer generation and
                        # all five LLM-judge providers (citation, completeness, root-cause,
                        # failure-category, evidence-chain) default to anthropic. `pytest` itself
                        # doesn't need a real key (unit tests mock generation/the judges).
pytest
```

To also run the frontend (query dashboard + trace/diff view, Streamlit):

```bash
pip install -e ".[dev,frontend]"
streamlit run src/frontend/app.py
```

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/PROJECT_SPEC.md](docs/PROJECT_SPEC.md).

## Build Phases

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Ingestion, Chunking, Hybrid Retrieval | Done |
| 2 | Generation with Citations | Done |
| 3 | Tracing & Instrumentation | Done (trace-per-request orchestrator wired for the query dashboard; a Phase 7 API-level orchestrator is still planned) |
| 4 | Backward Failure Analysis | Done |
| 5 | Visual Explorers & Frontend | Done |
| 6 | Evaluation Framework | In Progress |
| 7 | FastAPI, Docker, Portfolio Polish | Planned |

### Phase 1 Progress

| Component | Status |
|-----------|--------|
| Multi-format document loader (MD, TXT, HTML, PDF) | Done |
| Configurable chunking strategies (fixed_size, recursive_header, semantic) | Done |
| Embedding provider abstraction (`sentence_transformers` default, `openai` optional) | Done |
| Vector store provider abstraction (ChromaDB implemented, Qdrant planned) | Done |
| BM25 index | Done |
| Deduplication (cosine similarity, threshold 0.95) | Done |
| Dense retrieval (cosine top-k via ChromaDB) | Done |
| Sparse retrieval (BM25, score-normalized) | Done |
| RRF fusion (weighted, k=60) + HybridRetriever | Done |
| Voyage / Gemini / Cohere embedding providers | Done |
| Cross-encoder reranker (cuts to top 5) | Done |
| Cohere / Voyage reranker providers | Done |

### Phase 2 Progress

| Component | Status |
|-----------|--------|
| Grounded generation prompt (citation instructions, injection-hardened) | Done |
| Citation verification (LLM-as-judge per citation, Anthropic/OpenAI) | Done |
| Answer confidence scorer | Done |
| Graceful "I don't know" handling below confidence threshold | Done |

### Phase 3 Progress

| Component | Status |
|-----------|--------|
| Trace/Span pydantic data models | Done |
| Context manager / decorator to instrument pipeline steps | Done |
| Instrumentation applied across retrieval/generation pipeline | Done |
| Confidence scoring (1-5) populated on retrieval/ranking/verification/generation spans | Done |
| JSON trace file writer (`src/tracing/storage.py`) | Done |
| SQLite trace index (`src/tracing/index.py`) | Done |
| Trace-per-request orchestrator (`src/frontend/query_service.py`'s `ask_question`, driving the query dashboard) | Done |

### Phase 4 Progress

| Component | Status |
|-----------|--------|
| Backward root-cause span identification (`find_root_cause_span`, Anthropic/OpenAI step-quality judges) | Done |
| Failure-type categorization (`categorize_failure`, Anthropic/OpenAI failure-category judges) | Done |
| Narrative evidence-chain builder (`build_evidence_chain`, Anthropic/OpenAI evidence-chain judges) | Done |

### Phase 5 Progress

| Component | Status |
|-----------|--------|
| Trace view: color-coded flow graph, click-through span detail panel (`app_pages/trace_view.py`) | Done |
| On-demand root-cause diagnosis from the trace view (`diagnosis_service.py`) | Done |
| Diff view: received/produced/should-have-produced comparison, word-level divergence highlighting (`diff_panel.py`) | Done |
| Per-span human-correction persistence (`corrections.py`) | Done |
| Flagging interface: mark any trace "bad output" (including successful ones), confirm or override the root-cause diagnosis, persist the verdict for the eval loop (`flags.py`) | Done |
| Query dashboard: ask a question, see the cited answer, confidence breakdown, and hybrid-vs-dense-only retrieval toggle (`app_pages/query_dashboard.py`, `query_service.py`) | Done |
| Trace view + query dashboard combined into one `st.navigation` multipage app (`streamlit run src/frontend/app.py`) | Done |

### Phase 6 Progress

| Component | Status |
|-----------|--------|
| Golden Q&A dataset: 51 hand-written pairs (`data/golden/qa_dataset.json`) grounded in a fictional 8-doc corpus (`data/golden/corpus/`, since the real corpus is still empty) — lookup, multi-hop, no-answer, ambiguous, and edge-case categories | Done |
| Automated eval metrics: answer correctness + faithfulness (new LLM-as-judge modules), retrieval relevance (deterministic), citation accuracy (reuses `verify_citations`) — run via `python scripts/run_eval.py` (`src/evaluation/`) | Done |
| Chunking-strategy / retrieval-config comparison report | Planned |
| Auto-generate eval cases from confirmed human flags | Planned |
| Regression tracking | Planned |
| Failure analytics dashboard | Planned |
