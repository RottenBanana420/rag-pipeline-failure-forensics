# Architecture Overview

## 2026-06-28 — Phase 1 Scaffold

### Module Layout

```
src/
  config.py           # Central settings via pydantic-settings (singleton)
  ingestion/          # Document loaders, chunking strategies, deduplication
  retrieval/          # Dense (ChromaDB), sparse (BM25), RRF fusion, reranker
  generation/         # Grounded prompt, citation parser/verifier, confidence scorer
  tracing/            # Trace/Span models, context manager, decorator, JSON+SQLite writers
  analysis/           # Backward trace walker, failure categorizer, evidence chain builder
  evaluation/         # Golden dataset runner, metric calculators, regression tracker
  api/                # FastAPI app, route handlers
  frontend/           # Streamlit or React query dashboard and trace explorer
scripts/
  seed_corpus.py      # Index sample docs for local testing
  run_eval.py         # Execute full eval suite and print metrics
tests/
  unit/ingestion/     # Per-module unit tests
  unit/retrieval/
  integration/        # End-to-end pipeline tests against real ChromaDB
data/
  raw/                # Uploaded source documents
  processed/          # Normalized plaintext + metadata
  traces/             # JSON trace files (one per request)
  eval/               # Golden Q&A dataset and flagged failure cases
  chroma/             # ChromaDB file-based persistence
```

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

Every request is wrapped in a **Trace** (`trace_id`) containing **Spans** — one per pipeline step. Spans capture input, output, LLM prompt, token count, latency, and confidence score (1–5).

### Configuration

All runtime configuration flows through `src/config.py` (`Settings` class, pydantic-settings). Values are read from environment variables or `.env`. The module exposes a singleton `settings` object; tests instantiate `Settings()` directly to allow monkeypatching. See `.env.example` for the full variable reference.
