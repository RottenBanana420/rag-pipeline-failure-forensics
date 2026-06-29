# RAG Pipeline with Integrated Failure Forensics

Production-grade Retrieval-Augmented Generation system with built-in observability.
Every pipeline step is traced; failures are diagnosed automatically via backward span analysis.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # fill in OPENAI_API_KEY
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
| Embedding + ChromaDB storage | Done |
| BM25 index | Done |
| Deduplication (cosine similarity, threshold 0.95) | Done |
| Dense retrieval | Planned |
| Sparse retrieval | Planned |
| RRF fusion + reranker | Planned |
