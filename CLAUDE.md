# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Production-grade RAG (Retrieval-Augmented Generation) system with built-in observability. The system ingests internal documentation, retrieves context via hybrid search, generates grounded answers with inline citations, and traces every pipeline step to enable backward root-cause analysis when outputs degrade.

## Tech Stack

| Component | Choice |
|---|---|
| Language | Python 3.11+ |
| Embeddings | Pluggable via `EmbedderProtocol` — `sentence_transformers` (default, no API key), `openai` (`text-embedding-3-small`), `voyage` (`voyage-3.5`), `gemini` (`gemini-embedding-001`), or `cohere` (`embed-v4.0`); all API providers require their optional extra (`embed-openai`, `embed-voyage`, `embed-gemini`, `embed-cohere`) |
| Vector Store | Pluggable via `VectorStoreProtocol` — ChromaDB (file-based, implemented) or Qdrant (containerized, planned) |
| Sparse Search | `rank_bm25` |
| Reranking | Pluggable via `RerankerProtocol` — `sentence_transformers` cross-encoder (default, `cross-encoder/ms-marco-MiniLM-L6-v2`), `cohere` (`rerank-v4.0-pro`), or `voyage` (`rerank-2.5`); LLM-as-judge documented as a future provider |
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

# Add an optional embedding/store provider extra, e.g. OpenAI embeddings or Qdrant
pip install -e ".[dev,embed-openai]"

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
```

The following commands are part of the target end-state (later phases) and are **not runnable yet** — the underlying code is still a stub:

```bash
uvicorn src.api.main:app --reload --port 8000  # src/api/main.py not yet built (Phase 7)
docker compose up                                # no docker-compose.yml yet (Phase 7)
python scripts/seed_corpus.py                    # stub, raises NotImplementedError (Phase 1 follow-up)
python scripts/run_eval.py                        # stub, raises NotImplementedError (Phase 6)
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
  retrieval/          # EmbedderProtocol + make_embedder factory, VectorStoreProtocol +
                      # make_vector_store factory, ChromaVectorStore (cosine dedup + dimension
                      # guard), BM25Store, Indexer
                      # providers/           # One module per embedding provider (embedder_openai.py,
                      #                      # embedder_sentence_transformers.py, embedder_voyage.py,
                      #                      # embedder_gemini.py, embedder_cohere.py)
                      # DenseRetriever (cosine top-k), SparseRetriever (BM25 + score norm)
                      # VectorStoreHit (shared query result model)
                      # reciprocal_rank_fusion (RRF, k=60, weighted), HybridRetriever
                      # RerankerProtocol + make_reranker factory, SentenceTransformersReranker
                      # (providers/reranker_sentence_transformers.py) — cross-encoder second pass,
                      # cuts RRF's candidate pool (20) down to the final top-n (5)
                      # providers/reranker_cohere.py, providers/reranker_voyage.py — API-backed
                      # rerankers, same lazy-import factory pattern
  generation/         # Grounded prompt (prompts.py), citation_parser.py (regex [N] extraction),
                      # citation_verifier.py (CitationJudgeProtocol + make_citation_judge factory
                      # + verify_citations)
                      # providers/citation_judge_anthropic.py, providers/citation_judge_openai.py —
                      # LLM-as-judge citation verifiers, same lazy-import factory pattern
                      # confidence_scorer.py (CompletenessJudgeProtocol + make_completeness_judge
                      # factory + score_confidence — composite of retrieval confidence, citation
                      # coverage, and LLM-judged answer completeness)
                      # providers/completeness_judge_anthropic.py, providers/completeness_judge_openai.py —
                      # LLM-as-judge answer-completeness checkers, same lazy-import factory pattern
                      # fallback_response.py (build_fallback_response — deterministic, judge-free
                      # check of ConfidenceScore.retrieval_confidence against a threshold; returns a
                      # FallbackResponse with what was retrieved and which documents to check
                      # manually, or None if confidence is sufficient)
  tracing/            # models.py — Trace/Span pydantic models (PipelineStep/TraceStatus Literals)
                      # context.py — collect_spans() contextvar-based span sink
                      # instrumentation.py — span() context manager, traced() decorator,
                      # default_serialize() — applied across retrieval/generation (see below)
                      # JSON trace file writer, SQLite trace index, Trace-per-request
                      # orchestrator — planned
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

**Hybrid retrieval:** Dense (cosine similarity) + sparse (BM25) results are merged via Reciprocal Rank Fusion into a candidate pool, then a reranker cuts to the final top-n. Weights are configurable (default 0.7 dense / 0.3 sparse). See `docs/DECISIONS.md`.

**Reranking:** RRF's cutoff (`rerank_candidate_pool`, default 20) and the final answer size (`rerank_top_n`, default 5) are separate settings — a `model_validator` enforces `rerank_top_n <= rerank_candidate_pool`. The reranker is an injected, optional dependency on `HybridRetriever` (mirrors `Indexer`'s optional `embedder`/`vector_store`/`bm25_store` args): if `reranking_enabled=False` or no reranker is passed in, `retrieve()` falls back to slicing the RRF candidate pool directly. Chosen via `make_reranker(settings)` against `RerankerProtocol`, same lazy-import factory pattern as `make_embedder`. Three providers: `sentence_transformers` (local cross-encoder), `cohere` (`rerank-v4.0-pro`), and `voyage` (`rerank-2.5`) — the latter two call a hosted rerank API and overwrite each hit's `similarity` with the returned relevance score. Because Cohere's and Voyage's model names both start with `"rerank-"`, `make_reranker` can't use a prefix check (unlike `make_embedder`) to detect "user left `reranker_model` at its default" — it uses an exact-equality check against the sentence_transformers default (`cross-encoder/ms-marco-MiniLM-L6-v2`) instead, substituting the chosen provider's own default only when that default is still in place.

**Citation verification:** `verify_citations` (`src/generation/citation_verifier.py`) checks whether an LLM-generated answer's `[N]`-style citations are actually backed by the chunks they cite. `parse_citations` (`src/generation/citation_parser.py`) is a v1 regex heuristic — no sentence-boundary NLP — that finds contiguous `[N]` marker runs and pairs each with the claim text preceding it. For each parsed citation, `verify_citations` resolves the (1-indexed) chunk indices against the retrieved `VectorStoreHit`s; indices outside `1..len(hits)` are untrusted LLM output and short-circuit to an unsupported result without ever calling the judge. In-range citations get one `judge.judge(claim, evidence)` call each — no batching — via a `CitationJudgeProtocol` implementation chosen by `make_citation_judge(settings)` (same lazy-import factory pattern as `make_reranker`/`make_embedder`; `anthropic` or `openai`). The claim and evidence are wrapped in nonce-suffixed XML-style tags (`build_judge_prompt`, reusing `wrap_with_nonce` from the grounded-prompt module) so untrusted claim/evidence text can't forge a closing tag and break out of its block. This module is a standalone, directly-callable unit — the codebase has no generation orchestrator yet to wire it into automatically.

**Confidence scoring:** `score_confidence` (`src/generation/confidence_scorer.py`) rates a generated answer on three dimensions and combines them into one composite score. Retrieval confidence is the mean `similarity` across the hits used for generation (`0.0` if none). Citation coverage is the fraction of `verify_citations`' results with `supported=True` (`0.0` if none). Answer completeness comes from one `CompletenessJudgeProtocol.judge(question, answer)` call — an LLM-as-judge deciding whether every part of the question was addressed — chosen by `make_completeness_judge(settings)`, the same lazy-import factory pattern as `make_citation_judge`/`make_reranker`/`make_embedder` (providers: `anthropic`, `openai`). The three dimensions combine via a plain weighted sum (`confidence_retrieval_weight`/`confidence_citation_weight`/`confidence_completeness_weight`, default equal thirds, unnormalized) — the same convention `reciprocal_rank_fusion` uses for `dense_weight`/`sparse_weight`. Like citation verification, this is a standalone, directly-callable unit — the codebase has no generation orchestrator yet to wire it into automatically, and this task does not decide when a low score should feed into generation — that composition is a future orchestrator's job.

**Fallback response:** `build_fallback_response` (`src/generation/fallback_response.py`) checks `ConfidenceScore.retrieval_confidence` specifically (not the composite) against `settings.retrieval_confidence_threshold` (default `0.5`) — composite conflates citation/completeness quality with retrieval quality, and the spec calls out retrieval confidence by name. Below threshold (`retrieval_confidence < threshold`; at-or-above the threshold returns `None`, same `>=` convention `ChromaVectorStore` uses for its dedup check), it returns a frozen `FallbackResponse` (fixed `message`, a `retrieved_summary` line per hit, and deduped `documents_to_check` ranked by descending similarity, using `source_path` to disambiguate same-titled hits). No LLM call — deterministic, unlike the other two judge-backed dimensions. Same standalone-unit situation as citation verification and confidence scoring: no orchestrator exists yet to call this automatically.

**Chunking strategies:** Three switchable strategies — fixed-size with overlap (baseline), recursive character splitting on section headers (structure-aware), and semantic chunking on embedding similarity. Each chunk stores which strategy produced it.

**Deduplication:** Before inserting, cosine similarity is checked against existing chunks. Chunks with similarity > 0.95 are skipped.

**Provider abstraction:** Embedding and vector-store backends are chosen at runtime via `EMBEDDING_PROVIDER`/`VECTOR_STORE_PROVIDER` env vars, resolved through `make_embedder`/`make_vector_store` factories against `EmbedderProtocol`/`VectorStoreProtocol`. Provider SDKs are imported lazily inside the factory (not at module level) so installing one provider's extra doesn't require the others. On first write, `ChromaVectorStore` stamps the collection metadata with the embedder's `provider_id` and dimension count; on later opens it refuses to proceed if the configured provider doesn't match, instead of silently corrupting the index.

**Trace/Span data models:** `Span` and `Trace` (`src/tracing/models.py`) are the record the instrumentation below populates. `Span` has `span_id` (auto UUID4 hex), `step` (closed `Literal["ingestion", "retrieval", "ranking", "generation", "verification"]`), `input`/`output` (`str` — serialized by `traced()`/`span()`, not typed as the raw pipeline objects), `llm_prompt` (optional), `token_count` (optional, `>= 0`), `latency_ms` (`>= 0.0`), `confidence_score` (optional, `1-5` — left unset by every instrumented call today; no function in scope produces a discrete 1-5 rating, only continuous 0-1 floats), and `error` (optional, set when the instrumented call raised). `Trace` has `trace_id` (auto UUID4 hex), `spans` (defaults to `[]`), `final_output` (optional), and `status` (closed `Literal["success", "failure", "degraded"]`, required with no default). Both are plain (non-frozen) pydantic `BaseModel`s, matching `ProcessedDocument`/`Chunk` in `src/ingestion/models.py` rather than the frozen-dataclass convention used for judge-free result values (`FallbackResponse`, `ConfidenceScore`) — pydantic gives free JSON serialization, which the not-yet-built JSON/SQLite writers will need. No orchestrator yet assembles a `Trace` per request — see "Tracing instrumentation" below for what does exist.

**Tracing instrumentation:** `collect_spans()` (`src/tracing/context.py`) is a `ContextVar`-based sink; instrumented calls append their completed `Span` to whichever list is active, and are a no-op outside any `collect_spans()` block. `span(step, input)` (`src/tracing/instrumentation.py`) is a context manager for sites needing to attach LLM prompt/token/response detail mid-function — used directly inside all four judge providers' `judge()` methods (`citation_judge_anthropic.py`, `citation_judge_openai.py`, `completeness_judge_anthropic.py`, `completeness_judge_openai.py`), each with an `_extract_token_count` helper using `isinstance` checks (not try/except) since `MagicMock` auto-vivifies attributes rather than raising. `traced(step)` is a decorator built on `span()`, typed with `ParamSpec`/`TypeVar` so no `# type: ignore` is needed at any call site; applied to `DenseRetriever.retrieve`/`SparseRetriever.retrieve`/`reciprocal_rank_fusion` (`retrieval`), the three reranker providers' `rerank()` (`ranking`), `verify_citations` (`verification`), and `score_confidence`/`build_fallback_response` (`generation`). `HybridRetriever.retrieve` is deliberately not wrapped (its four leaf calls already are, and `Span` has no parent/child field). `verify_citations`/`score_confidence` keep their own wrapper span alongside their judges' inner spans — intentional, not redundant, since each has logic (citation short-circuits, arithmetic-only dimensions) a judge span never sees; see `docs/DECISIONS.md`.

**Backward root-cause analysis:** When a trace is flagged as failed, the system walks spans in reverse and uses an LLM-as-judge to score each step's output quality relative to its input. The first span with a significant quality drop is classified as the root cause. Failure types: Retrieval Failure, Ranking Failure, Extraction Hallucination, Citation Error, Generation Incomplete, Context Loss.

**Feedback loop:** Human-flagged bad outputs trigger automatic root-cause analysis. Confirmed diagnoses auto-generate new eval test cases (question, correct answer, failure category, failing step), growing the regression dataset over time.

### Confidence Scoring

Three dimensions reported per answer:
- **Retrieval confidence** — relevance of top-k chunks
- **Citation coverage** — percentage of claims with verified citations
- **Answer completeness** — whether all parts of the question were addressed

If retrieval confidence is below threshold, the system returns a structured "I don't know" response rather than hallucinating. See `build_fallback_response` in `src/generation/fallback_response.py`.

## Environment Variables

```
OPENAI_API_KEY=        # Required only if EMBEDDING_PROVIDER=openai, CITATION_JUDGE_PROVIDER=openai,
                        # or using GPT-4o for generation
ANTHROPIC_API_KEY=     # Required if using Claude Sonnet as LLM, or CITATION_JUDGE_PROVIDER=anthropic (default)
VOYAGE_API_KEY=        # Required only if EMBEDDING_PROVIDER=voyage
GEMINI_API_KEY=        # Required only if EMBEDDING_PROVIDER=gemini
COHERE_API_KEY=        # Required only if EMBEDDING_PROVIDER=cohere
EMBEDDING_PROVIDER=    # openai | sentence_transformers | voyage | gemini | cohere (default: sentence_transformers)
EMBEDDING_MODEL=       # Embedding model name (default: text-embedding-3-small; ignored by sentence_transformers unless set to a compatible model name)
EMBEDDING_DEVICE=      # auto | cpu | cuda | mps (default: auto; only affects sentence_transformers — API providers have no local device). "auto" lets the library auto-detect CUDA/MPS/CPU; the resolved device is logged at startup.
VECTOR_STORE_PROVIDER= # chroma | qdrant (default: chroma; qdrant not yet implemented)
CHROMA_PERSIST_DIR=    # Path for ChromaDB persistence (default: ./data/chroma)
RAW_DATA_DIR=          # Path for uploaded source documents (default: ./data/raw)
PROCESSED_DATA_DIR=    # Path for normalized plaintext + metadata (default: ./data/processed)
SQLITE_DB_PATH=        # Path for trace index (default: ./data/traces.db) — not active until Phase 3 (tracing) lands
TRACE_OUTPUT_DIR=      # Path for JSON trace files (default: ./data/traces/) — not active until Phase 3 (tracing) lands
LOG_LEVEL=             # DEBUG | INFO | WARNING (default: INFO)

# Retrieval
DENSE_TOP_K=           # Dense (cosine) candidates fetched before fusion (default: 10)
SPARSE_TOP_K=          # Sparse (BM25) candidates fetched before fusion (default: 10)
DENSE_WEIGHT=          # RRF weight for dense results (default: 0.7)
SPARSE_WEIGHT=         # RRF weight for sparse results (default: 0.3)
DEDUP_THRESHOLD=       # Cosine similarity above which an incoming chunk is skipped as a duplicate (default: 0.95)

# Chunking
CHUNK_STRATEGY=        # fixed_size | recursive_header | semantic (default: fixed_size)
CHUNK_SIZE=            # Characters per chunk (default: 1000, min: 100)
CHUNK_OVERLAP=         # Overlap between chunks (default: 200; must be < CHUNK_SIZE)
SEMANTIC_BREAKPOINT_PERCENTILE=  # Distance percentile threshold for semantic splits (default: 95.0)

# Reranking (cross-encoder second pass — precision boost after RRF fusion)
RERANK_CANDIDATE_POOL= # RRF's output size feeding the reranker (default: 20)
RERANK_TOP_N=          # Final number of chunks kept after reranking, or after RRF if reranking is disabled (default: 5; must be <= RERANK_CANDIDATE_POOL)
RERANKING_ENABLED=     # true | false (default: true) — when false, retrieve() falls back to slicing the RRF candidate pool directly
RERANKER_PROVIDER=     # sentence_transformers | cohere | voyage (default: sentence_transformers)
                        # cohere/voyage reuse the embed-cohere/embed-voyage extras (same SDKs) —
                        # no separate reranking extras
RERANKER_MODEL=        # Model name (default: cross-encoder/ms-marco-MiniLM-L6-v2 for sentence_transformers;
                        # rerank-v4.0-pro for cohere; rerank-2.5 for voyage — see make_reranker's
                        # model-default-substitution logic in src/retrieval/reranker.py)
RERANKER_DEVICE=       # auto | cpu | cuda | mps (default: auto; sentence_transformers only)

# Citation verification (LLM-as-judge check of [N]-style citations against cited chunks)
CITATION_JUDGE_PROVIDER=    # anthropic | openai (default: anthropic)
CITATION_JUDGE_MODEL=       # Model name (default: claude-sonnet-4-5 for anthropic; gpt-4o-2024-08-06 for openai)
CITATION_JUDGE_TEMPERATURE= # Sampling temperature for the judge call, 0.0-1.0 (default: 0.0)
ANSWER_COMPLETENESS_JUDGE_PROVIDER=    # anthropic | openai (default: anthropic)
ANSWER_COMPLETENESS_JUDGE_MODEL=       # Model name (default: claude-sonnet-4-5 for anthropic; gpt-4o-2024-08-06 for openai)
ANSWER_COMPLETENESS_JUDGE_TEMPERATURE= # Sampling temperature for the judge call, 0.0-1.0 (default: 0.0)

# Confidence scoring (composite of retrieval confidence, citation coverage, and answer completeness)
CONFIDENCE_RETRIEVAL_WEIGHT=    # Weight for retrieval confidence in the composite score (default: 0.3333...)
CONFIDENCE_CITATION_WEIGHT=     # Weight for citation coverage in the composite score (default: 0.3333...)
CONFIDENCE_COMPLETENESS_WEIGHT= # Weight for answer completeness in the composite score (default: 0.3333...)

# Fallback response (below this retrieval confidence, return a structured
# "insufficient information" response instead of generating an answer)
RETRIEVAL_CONFIDENCE_THRESHOLD= # Retrieval-confidence cutoff (default: 0.5)
```

## LLM Judge Cost Management

`CITATION_JUDGE_MODEL` and `ANSWER_COMPLETENESS_JUDGE_MODEL` default to `claude-sonnet-4-5` ($3/$15 per MTok in/out). For iterative dev/test runs, set both to `claude-haiku-4-5` (~5x cheaper on output, the dominant cost) — reserve Sonnet for occasional accuracy-checkpoint runs. A Claude Pro/Max subscription does **not** cover API usage; API calls need a separate console.anthropic.com key with its own billing.

**Testing cadence on real data:** run a small batch (10–20 questions) any time `generation/` changes or a new orchestrator step is wired in; run a fuller batch (100+) once per phase completion (end of Phase 2, 3, 4, 6). There's no CI spending real API money and no eval-runner yet (`scripts/run_eval.py` is a Phase 6 stub), so these are manual checkpoints, not per-commit gates.
