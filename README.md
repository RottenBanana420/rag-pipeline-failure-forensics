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
