# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Production-grade RAG (Retrieval-Augmented Generation) system with built-in observability. The system ingests internal documentation, retrieves context via hybrid search, generates grounded answers with inline citations, and traces every pipeline step to enable backward root-cause analysis when outputs degrade.

## Tech Stack

| Component | Choice |
|---|---|
| Language | Python 3.11+ |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector Store | ChromaDB (file-based) or Qdrant (containerized) |
| Sparse Search | `rank_bm25` |
| LLM | GPT-4o or Claude Sonnet (via API) |
| Chunking | LangChain text splitters |
| Tracing | OpenTelemetry + custom spans |
| Storage | SQLite (trace metadata) + JSON files (full traces) |
| API | FastAPI |
| Frontend | Streamlit or React |
| Containerization | Docker Compose |

## Error Resolution & Library Lookups

**Always use Context7 (or a web search) before implementing anything that touches a library or framework.** Do not rely on training knowledge — library APIs change, defaults shift, and version-specific bugs exist. Look up current docs first, then implement. This applies to new features, error debugging, API usage, and version migration. **Add a "Look up docs via Context7" step to every implementation plan before writing code.**

When encountering an error: look up the error message and library in Context7 or via web search before drawing any conclusions. Avoid assumptions about root cause.

## Commands

```bash
# Copy and fill in environment variables before running anything
cp .env.example .env

# Install dependencies (editable install with dev extras)
pip install -e ".[dev]"

# Run the API server
uvicorn src.api.main:app --reload --port 8000

# Run all tests
pytest

# Run a single test file
pytest tests/path/to/test_file.py -v

# Run tests matching a pattern
pytest -k "test_retrieval" -v

# Lint
ruff check src/ tests/
ruff format src/ tests/

# Type check
mypy src/

# Start full stack (API + ChromaDB + frontend)
docker compose up

# Seed the document corpus for testing
python scripts/seed_corpus.py

# Run evaluation suite
python scripts/run_eval.py
```

## Architecture

The system is built in seven phases. Each phase produces concrete, independently testable components.

### Pipeline Flow

```
Document → Ingestion → Chunking → Embedding → [ChromaDB | BM25 Index]
                                                        ↓
User Question → Embed → Dense Retrieval ─┐
                      → Sparse Retrieval ─┤→ RRF Fusion → Reranker → Top-5 Chunks
                                                                           ↓
                                                              LLM Generation + Citations
                                                                           ↓
                                                         Citation Verification + Confidence Score
                                                                           ↓
                                                                    Final Answer
```

Every request is wrapped in a **Trace** (unique `trace_id`) containing **Spans** — one per pipeline step. Spans capture input, output, LLM prompt, token count, latency, and a confidence score (1–5). Completed traces are written to JSON and indexed in SQLite.

### Module Layout

```
src/
  ingestion/          # Document loaders, chunking strategies, deduplication
  retrieval/          # Embedder, VectorStore (ChromaDB + cosine dedup), BM25Store, Indexer
                      # DenseRetriever (cosine top-k), SparseRetriever (BM25 + score norm)
                      # VectorStoreHit (shared query result model)
                      # reciprocal_rank_fusion (RRF, k=60, weighted), HybridRetriever
                      # Reranker: planned (cross-encoder, cuts RRF output to top 5)
  generation/         # Grounded prompt, citation parser, citation verifier, confidence scorer
  tracing/            # Trace/Span models, context manager, decorator, JSON + SQLite writers
  analysis/           # Backward trace walker, failure categorizer, evidence chain builder
  evaluation/         # Golden dataset runner, metric calculators, regression tracker
  api/                # FastAPI app, route handlers (/ask, /flag, /ingest, /documents)
  frontend/           # Streamlit or React query dashboard and trace explorer
scripts/
  seed_corpus.py      # Index sample docs for local testing
  run_eval.py         # Execute full eval suite and print metrics
tests/
  fixtures/           # Sample files (sample.md, sample.txt, sample.html) + PDF generator
  unit/ingestion/     # test_models, test_loader, test_storage, test_chunker
  unit/retrieval/     # test_embedder, test_vector_store, test_bm25_store, test_indexer
                      # test_dense_retriever, test_sparse_retriever
                      # test_fusion, test_hybrid_retriever
  integration/        # End-to-end pipeline tests against real ChromaDB
data/
  raw/                # Uploaded source documents
  processed/          # Normalized plaintext + metadata
  traces/             # JSON trace files (one per request)
  eval/               # Golden Q&A dataset and flagged failure cases
```

### Key Design Decisions

**Hybrid retrieval:** Dense (cosine similarity) + sparse (BM25) results are merged via Reciprocal Rank Fusion, then a cross-encoder reranker cuts to the top 5. Weights are configurable (default 0.7 dense / 0.3 sparse). See `docs/DECISIONS.md`.

**Chunking strategies:** Three switchable strategies — fixed-size with overlap (baseline), recursive character splitting on section headers (structure-aware), and semantic chunking on embedding similarity. Each chunk stores which strategy produced it.

**Deduplication:** Before inserting, cosine similarity is checked against existing chunks. Chunks with similarity > 0.95 are skipped.

**Tracing instrumentation:** A decorator pattern wraps any pipeline function in a span automatically. The span records the step name, serialized input/output, LLM prompt (if applicable), token count, latency, and confidence score. This means adding observability to a new step is one line of code.

**Backward root-cause analysis:** When a trace is flagged as failed, the system walks spans in reverse and uses an LLM-as-judge to score each step's output quality relative to its input. The first span with a significant quality drop is classified as the root cause. Failure types: Retrieval Failure, Ranking Failure, Extraction Hallucination, Citation Error, Generation Incomplete, Context Loss.

**Feedback loop:** Human-flagged bad outputs trigger automatic root-cause analysis. Confirmed diagnoses auto-generate new eval test cases (question, correct answer, failure category, failing step), growing the regression dataset over time.

### Confidence Scoring

Three dimensions reported per answer:
- **Retrieval confidence** — relevance of top-k chunks
- **Citation coverage** — percentage of claims with verified citations
- **Answer completeness** — whether all parts of the question were addressed

If retrieval confidence is below threshold, the system returns a structured "I don't know" response rather than hallucinating.

## Environment Variables

```
OPENAI_API_KEY=        # Required for embeddings and GPT-4o
ANTHROPIC_API_KEY=     # Required if using Claude Sonnet as LLM
EMBEDDING_MODEL=       # Embedding model name (default: text-embedding-3-small)
CHROMA_PERSIST_DIR=    # Path for ChromaDB persistence (default: ./data/chroma)
SQLITE_DB_PATH=        # Path for trace index (default: ./data/traces.db)
TRACE_OUTPUT_DIR=      # Path for JSON trace files (default: ./data/traces/)
LOG_LEVEL=             # DEBUG | INFO | WARNING (default: INFO)

# Chunking
CHUNK_STRATEGY=        # fixed_size | recursive_header | semantic (default: fixed_size)
CHUNK_SIZE=            # Characters per chunk (default: 1000, min: 100)
CHUNK_OVERLAP=         # Overlap between chunks (default: 200; must be < CHUNK_SIZE)
SEMANTIC_BREAKPOINT_PERCENTILE=  # Distance percentile threshold for semantic splits (default: 95.0)
```
