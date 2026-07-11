# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Production-grade RAG (Retrieval-Augmented Generation) system with built-in observability. The system ingests internal documentation, retrieves context via hybrid search, generates grounded answers with inline citations, and traces every pipeline step to enable backward root-cause analysis when outputs degrade.

## Tech Stack

| Component | Choice |
|---|---|
| Language | Python 3.11+ |
| Embeddings | Pluggable via `EmbedderProtocol` — `sentence_transformers` (default, no API key, extra `embed-local`), `openai` (`text-embedding-3-small`, extra `embed-openai`), `voyage` (`voyage-3.5`, extra `embed-voyage`), `gemini` (`gemini-embedding-001`, extra `embed-gemini`), or `cohere` (`embed-v4.0`, extra `embed-cohere`) |
| Vector Store | Pluggable via `VectorStoreProtocol` — ChromaDB (file-based, implemented, base dependency) or Qdrant (containerized, planned, extra `store-qdrant`) |
| Sparse Search | `rank_bm25` |
| Reranking | Pluggable via `RerankerProtocol` — `sentence_transformers` cross-encoder (default, `cross-encoder/ms-marco-MiniLM-L6-v2`), `cohere` (`rerank-v4.0-pro`), or `voyage` (`rerank-2.5`); LLM-as-judge documented as a future provider |
| LLM | GPT-4o or Claude Sonnet (via API, extra `llm-anthropic` for the Anthropic SDK — also covers the Anthropic citation/completeness judge providers) |
| Chunking | LangChain text splitters |
| Tracing | Custom span/trace system (`src/tracing/`) — `ContextVar`-based sink + pydantic `Span`/`Trace` models, not the `opentelemetry` SDK; no such dependency is declared (the `opentelemetry-*` entries in `requirements.txt` are a transitive dependency of `chromadb`'s own telemetry, unrelated to this project's tracing) |
| Storage | SQLite (trace metadata, `src/tracing/index.py`) + JSON files (full traces, `src/tracing/storage.py`) — implemented; no per-request orchestrator wires these into the pipeline automatically yet (see Module Layout) |
| API | FastAPI |
| Frontend | Streamlit (`src/frontend/`, implemented) — chosen over React because no API/HTTP layer exists yet; the trace view calls `load_trace`/`list_trace_records`/`find_root_cause_span`/etc. directly as Python functions instead of requiring `src/api/main.py` to be built first |
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

# Run the trace view (pipeline flow diagram, color-coded by span status)
streamlit run src/frontend/app.py

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
  config.py           # Settings (pydantic-settings) singleton — every env var below maps to a field here
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
  tracing/            # models.py — Trace/Span pydantic models (PipelineStep/TraceStatus Literals),
                      # Trace also carries timestamp (UTC, auto) and final_score (optional, unbounded)
                      # context.py — collect_spans() contextvar-based span sink
                      # instrumentation.py — span() context manager, traced() decorator,
                      # default_serialize(), confidence_from_score() — applied across
                      # retrieval/generation (see below)
                      # storage.py — save_trace()/load_trace(), one JSON file per trace_id
                      # index.py — SQLite metadata index (trace_id, timestamp, status, final_score,
                      # trace_path); init_trace_index(), index_trace(), get_trace_record(),
                      # list_trace_records()
                      # persistence.py — persist_trace(trace, settings), the standalone entry point
                      # tying storage.py + index.py together
                      # Trace-per-request orchestrator — planned (nothing yet calls collect_spans()
                      # and persist_trace() together for a live request)
  analysis/           # root_cause.py — StepQualityJudgeProtocol + make_step_quality_judge factory
                      # (same lazy-import pattern as make_citation_judge/make_completeness_judge) +
                      # find_root_cause_span, the backward span-quality walker
                      # failure_categorizer.py — FailureCategoryJudgeProtocol + make_failure_category_judge
                      # factory (same lazy-import pattern) + categorize_failure, which classifies a
                      # RootCauseDiagnosis into the failure taxonomy
                      # evidence_chain.py — EvidenceChainJudgeProtocol + make_evidence_chain_judge
                      # factory (same lazy-import pattern) + build_evidence_chain, which synthesizes
                      # a causal narrative from a RootCauseDiagnosis + FailureCategoryVerdict
                      # providers/         # step_quality_judge_anthropic.py, step_quality_judge_openai.py,
                      # failure_category_judge_anthropic.py, failure_category_judge_openai.py,
                      # evidence_chain_judge_anthropic.py, evidence_chain_judge_openai.py
  evaluation/         # Golden dataset runner, metric calculators, regression tracker
  api/                # FastAPI app, route handlers (/ask, /flag, /ingest, /documents)
  frontend/           # Streamlit trace view (`streamlit run src/frontend/app.py`)
                      # view_models.py — pure, Streamlit/LLM-independent: node_status (color-coding:
                      # root-cause span_id match -> red; confidence_score <= threshold -> yellow;
                      # else green) + build_graph_view_model (Trace -> TraceGraphViewModel, one node
                      # per span in trace.spans order, not one per distinct step name) +
                      # build_span_diff_view_model (Span + optional human-corrected expected output ->
                      # SpanDiffViewModel, a word-level difflib diff tagging each side's segments
                      # equal/expected_only/produced_only; segments are None when no correction exists)
                      # diagnosis_service.py — run_diagnosis, the on-demand (not automatic) root-cause
                      # pipeline: find_root_cause_span -> categorize_failure -> build_evidence_chain,
                      # short-circuiting the latter two when no root cause is found; the only
                      # src/frontend/ module importing from src/analysis/
                      # graph_render.py — streamlit_flow (streamlit-flow-component) adapter:
                      # TraceGraphViewModel -> StreamlitFlowNode/Edge, renders the graph, returns the
                      # clicked span_id
                      # detail_panel.py — node-detail panel: input/output/llm_prompt/confidence_score/
                      # latency_ms/token_count/error, plus an "embeddings not captured" note for
                      # retrieval/ranking-step spans (Span has no embeddings field)
                      # corrections.py — save_correction/load_correction, persists a human-entered
                      # per-span "expected output" as JSON, one file per trace_id keyed by span_id
                      # (data/eval/corrections/{trace_id}.json); not the Phase 6 golden dataset
                      # diff_panel.py — diff view: received/produced/"should have produced" side by
                      # side for a selected span in a non-success trace, computed live from the
                      # expected-output textbox's current value (not gated behind a save), with
                      # word-level divergence highlighting (build_span_diff_view_model's segments
                      # rendered as HTML spans via st.html — not st.markdown(unsafe_allow_html=True),
                      # which would run diff text through Markdown parsing first — wrapped in a
                      # white-space:pre-wrap container so whitespace-only divergence stays visible;
                      # each segment's text passed through html.escape() first since span/correction
                      # text is untrusted)
                      # app.py — Streamlit entrypoint wiring the above together
scripts/
  seed_corpus.py      # Index sample docs for local testing
  run_eval.py         # Execute full eval suite and print metrics
tests/
  fixtures/           # Sample files (sample.md, sample.txt, sample.html) + PDF generator
  unit/test_config.py # Settings defaults/validators
  unit/ingestion/     # test_models, test_loader, test_storage, test_chunker
  unit/retrieval/     # test_embedder, test_vector_store, test_bm25_store, test_indexer,
                      # test_dense_retriever, test_sparse_retriever, test_fusion,
                      # test_hybrid_retriever, test_models (confidence helper), factory tests
                      # providers/         # per-embedder and per-reranker provider tests
  unit/generation/    # test_prompts, test_citation_parser, test_citation_verifier,
                      # test_confidence_scorer, test_fallback_response
                      # providers/         # per-judge-provider tests (citation/completeness × anthropic/openai)
  unit/tracing/       # test_models, test_context, test_instrumentation, test_storage,
                      # test_index, test_persistence
  unit/analysis/      # test_root_cause, test_failure_categorizer, test_evidence_chain
                      # providers/         # per-judge-provider tests (step_quality/failure_category/
                      # evidence_chain_judge × anthropic/openai)
  unit/frontend/      # test_view_models (color-coding + view-model construction, including
                      # build_span_diff_view_model's diff-segment tagging), test_diagnosis_service
                      # (fake judges, no real API calls), test_corrections (save/load round-trip via
                      # tmp_path) — graph_render.py/detail_panel.py/diff_panel.py/app.py are
                      # Streamlit UI and verified manually instead (`streamlit run src/frontend/app.py`),
                      # not by automated tests
  integration/        # End-to-end pipeline tests against real ChromaDB
data/
  raw/                # Uploaded source documents
  processed/          # Normalized plaintext + metadata
  chroma/             # ChromaDB file-based persistence
  bm25_index.pkl      # Pickled BM25 corpus + chunk_ids
  traces/             # JSON trace files (one per request)
  eval/               # Golden Q&A dataset and flagged failure cases
```

### Key Design Decisions

**Hybrid retrieval:** Dense (cosine similarity) + sparse (BM25) results are merged via Reciprocal Rank Fusion into a candidate pool, then a reranker cuts to the final top-n. Weights are configurable (default 0.7 dense / 0.3 sparse). See `docs/DECISIONS.md`.

**Reranking:** RRF's cutoff (`rerank_candidate_pool`, default 20) and the final answer size (`rerank_top_n`, default 5) are separate settings — a `model_validator` enforces `rerank_top_n <= rerank_candidate_pool`. The reranker is an injected, optional dependency on `HybridRetriever` (mirrors `Indexer`'s optional `embedder`/`vector_store`/`bm25_store` args): if `reranking_enabled=False` or no reranker is passed in, `retrieve()` falls back to slicing the RRF candidate pool directly. Chosen via `make_reranker(settings)` against `RerankerProtocol`, same lazy-import factory pattern as `make_embedder`. Three providers: `sentence_transformers` (local cross-encoder), `cohere` (`rerank-v4.0-pro`), and `voyage` (`rerank-2.5`) — the latter two call a hosted rerank API and overwrite each hit's `similarity` with the returned relevance score. Because Cohere's and Voyage's model names both start with `"rerank-"`, `make_reranker` can't use a prefix check (unlike `make_embedder`) to detect "user left `reranker_model` at its default" — it uses an exact-equality check against the sentence_transformers default (`cross-encoder/ms-marco-MiniLM-L6-v2`) instead, substituting the chosen provider's own default only when that default is still in place.

**Citation verification:** `verify_citations` (`src/generation/citation_verifier.py`) checks whether an LLM-generated answer's `[N]`-style citations are actually backed by the chunks they cite. `parse_citations` (`src/generation/citation_parser.py`) is a v1 regex heuristic — no sentence-boundary NLP — that finds contiguous `[N]` marker runs and pairs each with the claim text preceding it. `GROUNDED_SYSTEM_PROMPT` instructs the model to place markers after the claim they support, but a model doesn't always comply (Haiku in particular will open a sentence with the marker instead, e.g. "According to the context, [1] the rotation is weekly"); when the text preceding a run is empty or ends with a comma — the structural signature of a lead-in with no claim yet — `parse_citations` treats it as a **leading marker** and scans forward instead, to the nearer of the next sentence-terminal punctuation (`.`/`!`/`?`) or the next citation run, merging that forward text with whatever (possibly empty) text preceded the marker. This is deliberately a punctuation-structure signal, not a lead-in-phrase blocklist, so it doesn't need to enumerate wordings like "according to". A known accepted edge case: if a leading marker's forward scan is capped by another citation run with no terminator in between, that next run's own preceding text is left empty and it too gets treated as leading — see `docs/DECISIONS.md`. For each parsed citation, `verify_citations` resolves the (1-indexed) chunk indices against the retrieved `VectorStoreHit`s; indices outside `1..len(hits)` are untrusted LLM output and short-circuit to an unsupported result without ever calling the judge. In-range citations get one `judge.judge(claim, evidence)` call each — no batching — via a `CitationJudgeProtocol` implementation chosen by `make_citation_judge(settings)` (same lazy-import factory pattern as `make_reranker`/`make_embedder`; `anthropic` or `openai`). The claim and evidence are wrapped in nonce-suffixed XML-style tags (`build_judge_prompt`, reusing `wrap_with_nonce` from the grounded-prompt module) so untrusted claim/evidence text can't forge a closing tag and break out of its block. This module is a standalone, directly-callable unit — the codebase has no generation orchestrator yet to wire it into automatically.

**Confidence scoring:** `score_confidence` (`src/generation/confidence_scorer.py`) rates a generated answer on three dimensions and combines them into one composite score. Retrieval confidence is the mean `similarity` across the hits used for generation (`0.0` if none). Citation coverage is the fraction of `verify_citations`' results with `supported=True` (`0.0` if none). Answer completeness comes from one `CompletenessJudgeProtocol.judge(question, answer)` call — an LLM-as-judge deciding whether every part of the question was addressed — chosen by `make_completeness_judge(settings)`, the same lazy-import factory pattern as `make_citation_judge`/`make_reranker`/`make_embedder` (providers: `anthropic`, `openai`). The three dimensions combine via a plain weighted sum (`confidence_retrieval_weight`/`confidence_citation_weight`/`confidence_completeness_weight`, default equal thirds, unnormalized) — the same convention `reciprocal_rank_fusion` uses for `dense_weight`/`sparse_weight`. Like citation verification, this is a standalone, directly-callable unit — the codebase has no generation orchestrator yet to wire it into automatically, and this task does not decide when a low score should feed into generation — that composition is a future orchestrator's job.

**Fallback response:** `build_fallback_response` (`src/generation/fallback_response.py`) checks `ConfidenceScore.retrieval_confidence` specifically (not the composite) against `settings.retrieval_confidence_threshold` (default `0.5`) — composite conflates citation/completeness quality with retrieval quality, and the spec calls out retrieval confidence by name. Below threshold (`retrieval_confidence < threshold`; at-or-above the threshold returns `None`, same `>=` convention `ChromaVectorStore` uses for its dedup check), it returns a frozen `FallbackResponse` (fixed `message`, a `retrieved_summary` line per hit, and deduped `documents_to_check` ranked by descending similarity, using `source_path` to disambiguate same-titled hits). No LLM call — deterministic, unlike the other two judge-backed dimensions. Same standalone-unit situation as citation verification and confidence scoring: no orchestrator exists yet to call this automatically.

**Chunking strategies:** Three switchable strategies — fixed-size with overlap (baseline), recursive character splitting on section headers (structure-aware), and semantic chunking on embedding similarity. Each chunk stores which strategy produced it.

**Deduplication:** Before inserting, cosine similarity is checked against existing chunks. Chunks with similarity > 0.95 are skipped.

**Provider abstraction:** Embedding and vector-store backends are chosen at runtime via `EMBEDDING_PROVIDER`/`VECTOR_STORE_PROVIDER` env vars, resolved through `make_embedder`/`make_vector_store` factories against `EmbedderProtocol`/`VectorStoreProtocol`. Provider SDKs are imported lazily inside the factory (not at module level) so installing one provider's extra doesn't require the others. On first write, `ChromaVectorStore` stamps the collection metadata with the embedder's `provider_id` and dimension count; on later opens it refuses to proceed if the configured provider doesn't match, instead of silently corrupting the index.

**Trace/Span data models:** `Span` and `Trace` (`src/tracing/models.py`) are the record the instrumentation below populates. `Span` has `span_id` (auto UUID4 hex), `step` (closed `Literal["ingestion", "retrieval", "ranking", "generation", "verification", "analysis"]` — `"analysis"` added alongside the backward root-cause analysis feature below, for the step-quality judge's own spans), `input`/`output` (`str` — serialized by `traced()`/`span()`, not typed as the raw pipeline objects), `llm_prompt` (optional), `token_count` (optional, `>= 0`), `latency_ms` (`>= 0.0`), `confidence_score` (optional, `1-5` — populated on most instrumented spans via `confidence_from_score()`, which linearly maps that step's own continuous 0-1 quality signal onto the 1-5 scale; left `None` on spans with no such signal, e.g. `build_fallback_response`), `error` (optional, set when the instrumented call raised), and `is_gate` (`bool`, default `False` — marks a deterministic post-processing/gating span that mechanically transforms already-computed upstream signals rather than performing new pipeline work; see "Backward root-cause span identification" below and `docs/DECISIONS.md`). `Trace` has `trace_id` (auto UUID4 hex), `timestamp` (UTC-aware `datetime`, auto `default_factory`), `spans` (defaults to `[]`), `final_output` (optional), `status` (closed `Literal["success", "failure", "degraded"]`, required with no default), and `final_score` (optional `float`, no `ge`/`le` bounds — `ConfidenceScore.composite`'s three weights are independently configurable and not validated to sum to 1, so the composite it's usually populated from isn't guaranteed to land in `[0,1]`). Both `Span`/`Trace` are plain (non-frozen) pydantic `BaseModel`s, matching `ProcessedDocument`/`Chunk` in `src/ingestion/models.py` rather than the frozen-dataclass convention used for judge-free result values (`FallbackResponse`, `ConfidenceScore`) — pydantic gives free JSON serialization, which the JSON/SQLite writers below use directly. No orchestrator yet assembles a `Trace` per request — see "Tracing instrumentation" below for what does exist.

**Trace persistence:** `src/tracing/storage.py`'s `save_trace(trace, output_dir)`/`load_trace(trace_id, output_dir)` read/write one `{trace_id}.json` file per trace via `model_dump_json(indent=2)`/`model_validate_json()` — the same convention as `src/ingestion/storage.py`'s `save_processed`/`load_processed`, but deliberately without that function's `shutil.rmtree`: each trace is an independent file, and writing a new trace must never delete previously written ones. `src/tracing/index.py` maintains a SQLite metadata index (raw stdlib `sqlite3`, no ORM) with one `traces` table (`trace_id` PK, `timestamp`, `status`, `final_score`, and `trace_path` — the last one isn't in the task's literal column list but is required to actually resolve a metadata row back to its full JSON trace) via `init_trace_index()`, `index_trace()` (`INSERT OR REPLACE`, idempotent per `trace_id`), `get_trace_record()`, and `list_trace_records()` (filterable by `status`, ordered by `timestamp` desc). `src/tracing/persistence.py`'s `persist_trace(trace, settings)` is the standalone entry point tying the two together — same "directly-callable unit, no orchestrator" pattern as `verify_citations`/`score_confidence`/`build_fallback_response`. If the JSON write succeeds but SQLite indexing then fails, the exception propagates uncaught and the JSON file is left in place (it's the durable source of truth; a missing index row is repairable later by re-running `index_trace`, an already-deleted trace file is not). Settings gained `trace_output_dir` (default `./data/traces`) and `sqlite_db_path` (default `./data/traces.db`), matching the `raw_data_dir`/`processed_data_dir` convention.

**Tracing instrumentation:** `collect_spans()` (`src/tracing/context.py`) is a `ContextVar`-based sink; instrumented calls append their completed `Span` to whichever list is active, and are a no-op outside any `collect_spans()` block. `span(step, input)` (`src/tracing/instrumentation.py`) is a context manager for sites needing to attach LLM prompt/token/response detail mid-function — used directly inside all six judge providers' `judge()` methods (`citation_judge_anthropic.py`, `citation_judge_openai.py`, `completeness_judge_anthropic.py`, `completeness_judge_openai.py`, `step_quality_judge_anthropic.py`, `step_quality_judge_openai.py`), each with an `_extract_token_count` helper using `isinstance` checks (not try/except) since `MagicMock` auto-vivifies attributes rather than raising, and inside `reciprocal_rank_fusion` (`retrieval`) to compute its own confidence score from pre-fusion hit similarity (see below). `traced(step, confidence_fn=None, is_gate=False)` is a decorator built on `span()`, typed with `ParamSpec`/`TypeVar` so no `# type: ignore` is needed at any call site; applied to `DenseRetriever.retrieve`/`SparseRetriever.retrieve` (`retrieval`), the three reranker providers' `rerank()` (`ranking`), `verify_citations` (`verification`), and `score_confidence`/`build_fallback_response` (`generation`). `HybridRetriever.retrieve` is deliberately not wrapped (its four leaf calls already are, and `Span` has no parent/child field). `verify_citations`/`score_confidence` keep their own wrapper span alongside their judges' inner spans — intentional, not redundant, since each has logic (citation short-circuits, arithmetic-only dimensions) a judge span never sees; see `docs/DECISIONS.md`. `is_gate=True` marks the recorded span `Span.is_gate=True`; `score_confidence`/`build_fallback_response` are the only two call sites that pass it, since both are deterministic pass-throughs over already-computed upstream signals (see "Backward root-cause span identification" below and `docs/DECISIONS.md`).

**Confidence score population:** `confidence_from_score(value)` (`src/tracing/instrumentation.py`) linearly maps a continuous `0-1` signal onto `Span.confidence_score`'s `1-5` scale (clamping first, so out-of-range inputs can't violate the field's `ge=1, le=5` constraint). `traced()`'s optional `confidence_fn` parameter is called with the wrapped function's return value once it succeeds, and its result populates `s.confidence_score`; typed `Callable[[Any], int | None]` rather than against `traced()`'s own `T`, because binding `T` from `confidence_fn` too causes mypy to mis-solve `T` before seeing the decorator applied (see `docs/DECISIONS.md`). `mean_similarity_confidence` (`src/retrieval/models.py`) is the shared `confidence_fn` for `DenseRetriever.retrieve`, `SparseRetriever.retrieve`, and all three rerankers' `rerank()` — mean `similarity` across returned hits, `None` if empty. `reciprocal_rank_fusion` doesn't use `@traced`'s `confidence_fn` hook: it still uses `span()` directly and calls `mean_similarity_confidence` itself, inside the function, on `result` — the RRF fused score (`~1/60` scale, used only to select and order the `top_n` candidates) is never written onto a returned hit's `similarity`, so `result` already carries each hit's real pre-fusion similarity (dense cosine, or sparse max-normalized BM25) and needs no separate pre-fusion lookup the way it once did (see `docs/DECISIONS.md`, 2026-07-09 entry, for why the earlier RRF-score-as-`similarity` behavior was reversed). `verify_citations` derives confidence from the fraction of citations verified `supported=True`; `score_confidence` derives it from `ConfidenceScore.composite`. `build_fallback_response` sets no confidence score — its result is a threshold gate, not a graded signal.

**Backward root-cause span identification:** `find_root_cause_span` (`src/analysis/root_cause.py`) walks a `Trace`'s spans in reverse execution order (`Trace.spans` is a flat list — `Span` has no parent/child field), calling one `StepQualityJudgeProtocol.judge(step, input, output)` per span to score its input→output transformation quality on a 1-5 scale (same scale as `Span.confidence_score`), chosen by `make_step_quality_judge(settings)` — same lazy-import factory pattern as `make_citation_judge`/`make_completeness_judge` (providers: `anthropic`, `openai`). The judge prompt is step-aware: `STEP_QUALITY_CRITERIA` gives each `PipelineStep` its own criteria for what a "reasonable transformation" means, so the judge doesn't apply generation criteria to a ranking step or vice versa. A span scoring at or below `settings.root_cause_quality_threshold` (default `2`) is "unreasonable"; the walk stops at the first (walking backward) span scoring *above* the threshold — that span is healthy and marks the boundary of the failing run. Spans with `Span.is_gate=True` (`score_confidence`, `build_fallback_response`) are skipped entirely instead of being judged: no judge call, no candidate update, never able to end the walk. Gate spans mechanically transform already-computed upstream signals and are therefore internally self-consistent by construction — a judge would score one "reasonable" regardless of whether the upstream data it received was already corrupted, so treating one as a healthy boundary could prematurely mask a genuinely corrupted upstream span; see `docs/DECISIONS.md`. The root cause is the **earliest** span in the contiguous unhealthy tail (the last one remembered before the healthy boundary, or before spans run out), not simply the last-executed bad span — a cascading failure should surface where the corruption originated, not its downstream symptoms. Only the unhealthy tail is judged; spans before an already-healthy boundary — and all gate spans — are never called. Returns `None` if no non-gate span is ever at/below threshold, mirroring `build_fallback_response`'s `Optional`-return convention for "nothing wrong here." The step-quality judge's own LLM call uses a sixth `PipelineStep` value, `"analysis"`, for its span — it doesn't belong to any of the five original pipeline steps since it runs later, over an already-completed trace. Like every other generation-module feature, this is a standalone, directly-callable unit: `find_root_cause_span` takes an already-loaded `Trace` and a judge instance as plain parameters; no orchestrator yet loads a flagged trace and calls this automatically. Failure-type categorization is implemented separately, in `failure_categorizer.py` (see below); the narrative evidence-chain builder is implemented separately too, in `evidence_chain.py` (see below).

**Failure categorization:** `categorize_failure` (`src/analysis/failure_categorizer.py`) classifies a `RootCauseDiagnosis` (the output of `find_root_cause_span`) into the failure taxonomy: Retrieval Failure, Ranking Failure, Extraction Hallucination, Citation Error, Generation Incomplete, or Context Loss. The mapping from `Span.step` to a category isn't 1:1 — a `"generation"`-step root cause could be any of three categories (Extraction Hallucination, Generation Incomplete, Context Loss) — so classification is delegated to one `FailureCategoryJudgeProtocol.classify(step, input, output, quality_rationale)` call per diagnosis, chosen by `make_failure_category_judge(settings)` — same lazy-import factory pattern as `make_step_quality_judge` (providers: `anthropic`, `openai`). `STEP_TO_PLAUSIBLE_CATEGORIES` restricts the judge to the category subset that's actually plausible for the root-cause span's step (e.g. a `"verification"`-step root cause can only be a Citation Error), stated explicitly in the prompt as a guardrail against invalid picks. The six-category taxonomy has no bucket for `"ingestion"`- or `"analysis"`-step root causes, so `FailureCategory` adds a 7th value, `"other"`, covering both — without it, `categorize_failure` would have no valid answer for a legitimately-returned diagnosis whose root cause is an ingestion-step span. `quality_rationale` passes along the step-quality judge's own explanation (`RootCauseDiagnosis.rationale`) as extra classification signal. Same standalone-unit convention as `find_root_cause_span`: `categorize_failure` takes an already-computed `RootCauseDiagnosis` and a judge instance as plain parameters, and adds no span of its own — only the provider's `classify()` call emits a `step="analysis"` span.

**Evidence chain narrative:** `build_evidence_chain` (`src/analysis/evidence_chain.py`) synthesizes `EvidenceChain.narrative` — a structured causal explanation like "Retrieval ranked the most relevant chunk at position 7 instead of position 1. This propagated to Generation, which selected from the top 5 and missed the answer" — from a `RootCauseDiagnosis` (from `find_root_cause_span`) and a `FailureCategoryVerdict` (from `categorize_failure`, taken as an already-computed parameter, mirroring `categorize_failure`'s own "takes upstream results as plain parameters" convention). Synthesis is delegated to one `EvidenceChainJudgeProtocol.narrate(category, category_rationale, chain)` call per diagnosis, chosen by `make_evidence_chain_judge(settings)` — same lazy-import factory pattern as `make_failure_category_judge`/`make_step_quality_judge` (providers: `anthropic`, `openai`). An LLM-as-judge was chosen over a deterministic string template because a template mechanically concatenating each span's already-isolated per-span rationale can't produce genuine cross-span causal reasoning ("this propagated to X, which then...") — each existing rationale judges its own span in isolation, never the relationship between spans; every other qualitative synthesis in this codebase (citation verification, completeness, step quality, failure categorization) is likewise judge-backed. `RootCauseDiagnosis.evaluated_spans` is last-executed-first (reverse-walk order); `build_evidence_chain` reverses it into chronological, root-cause-first order before building the `chain`, so both the judge prompt and the returned `EvidenceChain.evidence` read in execution order. The protocol takes `list[EvidenceEntry]` — a new, purpose-built, flat dataclass (`step`/`input`/`output`/`score`/`rationale`) owned by `evidence_chain.py` — rather than `root_cause.py`'s `SpanQualityResult`, keeping provider implementations decoupled from that module's dataclasses, the same rationale already used for `FailureCategoryJudgeProtocol.classify` taking scalars instead of `RootCauseDiagnosis` itself. `EvidenceChainVerdict` has a single `narrative` field, unlike every other verdict's decision-plus-rationale shape — here the narrative already is the explanation, so a separate rationale field would be redundant. The prompt builder (`build_evidence_chain_judge_prompt`) wraps an unbounded number of per-entry blocks in nonce-suffixed tags sharing one nonce per call, using indexed tag names (`span-{i}-input`/`span-{i}-output`/`span-{i}-rationale`) rather than the fixed 2-3 named blocks every earlier judge prompt used — `step`/`score` are safe (a closed `Literal` and a `1-5`-bounded `int`) and appear unwrapped. Adds no span of its own, same convention as `find_root_cause_span`/`categorize_failure`.

**Trace view (frontend):** `src/frontend/app.py` (`streamlit run src/frontend/app.py`) is a Streamlit app rendering a `Trace`'s spans as a left-to-right flow diagram, color-coded green/yellow/red, with a click-through node-detail panel. Streamlit was chosen over React specifically because no API/HTTP layer exists yet (`src/api/` is still a Phase 7 placeholder) — the app calls `load_trace`/`list_trace_records`/`find_root_cause_span`/`categorize_failure`/`build_evidence_chain` directly as Python functions, versus React which would require standing up `src/api/main.py` from scratch first, turning "build a trace view" into "build Phase 5 and Phase 7 at once." `Trace.spans` is a flat, execution-order list with no parent/child field, and multiple spans can share the same `step` (e.g. `HybridRetriever`'s dense/sparse retrieval legs are both `step="retrieval"`), so `build_graph_view_model` (`src/frontend/view_models.py`) makes **each span its own node**, positioned in `trace.spans` order and connected by sequential edges — not one node per distinct step name. `node_status` colors a node red if it matches the current `RootCauseDiagnosis.root_cause_span.span_id` (always wins, regardless of `confidence_score`), else yellow if `Span.confidence_score <= settings.root_cause_quality_threshold` (reusing that setting's existing 1-5 "at or below is unreasonable" semantics rather than inventing a parallel threshold), else green — this needs no LLM call, so opening any trace is free. Root-cause coloring only appears after an explicit "Diagnose root cause" button click, which calls `run_diagnosis` (`src/frontend/diagnosis_service.py`, the only `src/frontend/` module importing from `src/analysis/`) — real per-span LLM judge spend, per the "LLM Judge Cost Management" section below, so it must be deliberate, not automatic on trace load. The resulting `DiagnosisResult` is cached in `st.session_state` keyed by `trace_id`, so revisiting the same trace within a session doesn't re-spend. The node graph is rendered via `streamlit_flow` (`streamlit-flow-component`, a new `frontend` extra in `pyproject.toml`), chosen via Context7 lookup over `streamlit-agraph` (undocumented there — a maintenance-signal red flag); its `get_node_on_click`/`StreamlitFlowState.selected_id` click detection, per-node `style` dict, and `ManualLayout` positioning map directly onto this flat/ordered span-list model. `Span` has no embeddings/vector field, so `detail_panel.py` shows an explicit "embeddings not captured in this trace" note for retrieval/ranking-step spans rather than extending `Span` — that would mean touching tracing instrumentation across every retrieval/ranking provider and storing potentially large vectors in every trace JSON, out of scope for a UI-only task. See `docs/DECISIONS.md` for the streamlit_flow `key=`/`st.session_state` collision gotcha hit during implementation.

**Diff view (frontend):** `docs/PROJECT_SPEC.md` (Phase 5, item 2) asks for a per-span "received vs. produced vs. should have produced" comparison on failed traces, with the divergence highlighted. No golden-dataset or human-correction data source exists yet — `src/evaluation/` is still a Phase 6 placeholder and no `Span`/`Trace` field carries an expected output — so `src/frontend/corrections.py` introduces the "human correction" half of that spec sentence only: `save_correction`/`load_correction` persist a human-typed expected output per `(trace_id, span_id)` as one JSON file per trace under `settings.human_corrections_dir` (default `./data/eval/corrections`), mirroring `src.tracing.storage`'s one-file-per-id convention. This is deliberately not the full Phase 6 golden dataset (50+ hand-written Q&A pairs plus automated eval metrics) — that remains out of scope here; a future Phase 6 orchestrator can read the same correction files. `build_span_diff_view_model` (`src/frontend/view_models.py`) computes the divergence with a word-level diff (`difflib.SequenceMatcher` over whitespace-preserving tokens, the same technique `difflib.HtmlDiff`/`git diff --word-diff` use — no new dependency), returning `None` segments when no correction has been entered yet rather than diffing against an empty string. `diff_panel.py` builds this from the expected-output text area's *live* return value, not the on-disk saved correction, so the highlighted divergence updates immediately as the user types/blurs rather than requiring a save first; "Save correction" (`st.toast("Correction saved.", icon="✅")` on click, no `st.rerun()`) is purely for persistence. It renders the three columns and, only for `trace.status != "success"` traces (matching the spec's "for failed traces" and the existing "Diagnose root cause" button's own success-gate), the highlighted segments as inline-styled HTML via `st.html()` (not `st.markdown(unsafe_allow_html=True)` — that would run the diff text through Markdown parsing first, corrupting literal `*`/`_` characters), wrapped in a `white-space:pre-wrap` container so whitespace-only divergence doesn't visually collapse. Every segment's text is passed through `html.escape()` first, since `Span.output` and human corrections are both untrusted text and neither `st.markdown(unsafe_allow_html=True)` nor `st.html` escapes automatically (confirmed via Context7 lookup against Streamlit's docs); `detail_panel.py` avoids this risk entirely by using `st.text_area`, which auto-escapes, but the diff view needs per-word span-level styling that a plain text widget can't produce. `st.html` predates the project's prior `streamlit>=1.38` floor, so `pyproject.toml`'s `frontend` extra now requires `streamlit>=1.41`.

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
SQLITE_DB_PATH=        # Path for the SQLite trace metadata index (default: ./data/traces.db)
TRACE_OUTPUT_DIR=      # Directory for per-trace JSON files (default: ./data/traces/)
HUMAN_CORRECTIONS_DIR= # Directory for per-span human-entered "expected output" corrections used by
                        # the trace view's diff view (default: ./data/eval/corrections)
LOG_LEVEL=             # DEBUG | INFO | WARNING | ERROR | CRITICAL (default: INFO)

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

# Root-cause analysis (backward span-quality judging — LLM-as-judge scores each
# span's input→output transformation quality, 1-5, walking a failed trace backward)
ROOT_CAUSE_JUDGE_PROVIDER=    # anthropic | openai (default: anthropic)
ROOT_CAUSE_JUDGE_MODEL=       # Model name (default: claude-sonnet-4-5 for anthropic; gpt-4o-2024-08-06 for openai)
ROOT_CAUSE_JUDGE_TEMPERATURE= # Sampling temperature for the judge call, 0.0-1.0 (default: 0.0)
ROOT_CAUSE_QUALITY_THRESHOLD= # A span scoring at or below this (1-5) is treated as an unreasonable transformation (default: 2)

# Failure categorization (classifies a root-cause span into the failure
# taxonomy — Retrieval Failure, Ranking Failure, Extraction Hallucination,
# Citation Error, Generation Incomplete, Context Loss, or Other)
FAILURE_CATEGORY_JUDGE_PROVIDER=    # anthropic | openai (default: anthropic)
FAILURE_CATEGORY_JUDGE_MODEL=       # Model name (default: claude-sonnet-4-5 for anthropic; gpt-4o-2024-08-06 for openai)
FAILURE_CATEGORY_JUDGE_TEMPERATURE= # Sampling temperature for the judge call, 0.0-1.0 (default: 0.0)

# Evidence chain narrative (LLM-as-judge synthesizes a causal narrative from
# the ordered evidence chain leading to a diagnosed root cause)
EVIDENCE_CHAIN_JUDGE_PROVIDER=    # anthropic | openai (default: anthropic)
EVIDENCE_CHAIN_JUDGE_MODEL=       # Model name (default: claude-sonnet-4-5 for anthropic; gpt-4o-2024-08-06 for openai)
EVIDENCE_CHAIN_JUDGE_TEMPERATURE= # Sampling temperature for the judge call, 0.0-1.0 (default: 0.0)
```

## LLM Judge Cost Management

`CITATION_JUDGE_MODEL`, `ANSWER_COMPLETENESS_JUDGE_MODEL`, `ROOT_CAUSE_JUDGE_MODEL`, `FAILURE_CATEGORY_JUDGE_MODEL`, and `EVIDENCE_CHAIN_JUDGE_MODEL` all default to `claude-sonnet-4-5` ($3/$15 per MTok in/out). For iterative dev/test runs, set all five to `claude-haiku-4-5` (~5x cheaper on output, the dominant cost) — reserve Sonnet for occasional accuracy-checkpoint runs. Root-cause and failure-category judges run once per span/diagnosis during a backward trace walk, so they add up the same way the per-citation and per-answer judges do. The evidence-chain judge runs once per diagnosis (same frequency as the failure-category judge), but its prompt payload scales with the full unhealthy tail's input/output text rather than a single span, making it the most token-heavy of the five per call despite running least often. A Claude Pro/Max subscription does **not** cover API usage; API calls need a separate console.anthropic.com key with its own billing.

**Testing cadence on real data:** run a small batch (10–20 questions) any time `generation/` changes or a new orchestrator step is wired in; run a fuller batch (100+) once per phase completion (end of Phase 2, 3, 4, 6). There's no CI spending real API money and no eval-runner yet (`scripts/run_eval.py` is a Phase 6 stub), so these are manual checkpoints, not per-commit gates.
