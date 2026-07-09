# RAG Pipeline with Integrated Failure Forensics

Production-grade Retrieval-Augmented Generation system with built-in observability.
Every pipeline step is traced; failures are diagnosed automatically via backward span analysis.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # defaults work out of the box (local sentence_transformers embeddings, no API key needed)
pytest
```

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/PROJECT_SPEC.md](docs/PROJECT_SPEC.md).

## Build Phases

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Ingestion, Chunking, Hybrid Retrieval | Done |
| 2 | Generation with Citations | In Progress |
| 3 | Tracing & Instrumentation | In Progress |
| 4 | Backward Failure Analysis | In Progress |
| 5 | Visual Explorers & Frontend | Planned |
| 6 | Evaluation Framework | Planned |
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
| Trace-per-request orchestrator | Planned |

### Phase 4 Progress

| Component | Status |
|-----------|--------|
| Backward root-cause span identification (`find_root_cause_span`, Anthropic/OpenAI step-quality judges) | Done |
| Failure-type categorization (`categorize_failure`, Anthropic/OpenAI failure-category judges) | Done |
| Narrative evidence-chain builder (`build_evidence_chain`, Anthropic/OpenAI evidence-chain judges) | Done |
