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
| 1 | Ingestion, Chunking, Hybrid Retrieval | In progress |
| 2 | Generation with Citations | Planned |
| 3 | Tracing & Instrumentation | Planned |
| 4 | Backward Failure Analysis | Planned |
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
| Cross-encoder reranker (cuts to top 5) | Planned |
