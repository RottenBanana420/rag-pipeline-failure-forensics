# Architecture Decision Records

## 2026-07-04 — Span Instrumentation

**`ContextVar`-based sink (`collect_spans()`), not a required `Trace` object per call** — No orchestrator exists yet to assemble one `Trace` per request (every generation-module function is still "a standalone, directly-callable unit" per the entries above). Requiring every instrumented call to build or receive a `Trace` would invent an API shape for an orchestrator that doesn't exist. `collect_spans()` instead yields a plain `list[Span]` that a future orchestrator wraps into a `Trace` itself; outside any `collect_spans()` block, instrumented calls run exactly as before, which is what keeps every pre-existing unit test (none of which set up tracing) passing unmodified.

**`span()` context manager for LLM call sites, `traced()` decorator for everything else** — A decorator can only see a function's arguments and return value, but the LLM prompt, raw response, and token usage a judge provider's `judge()` method produces exist only *inside* the method body, after the API call returns. `span()` is the primitive that supports attaching that mid-function detail (`s.llm_prompt = ...`, `s.token_count = ...`); `traced()` is sugar over `span()` for the common case where a function's arguments and return value are the whole story.

**Token counts extracted via `isinstance` checks, not `try`/`except`** — `unittest.mock.MagicMock` auto-vivifies any attribute access (`mock.usage.input_tokens` returns another `MagicMock`, not an `AttributeError`), so every existing judge-provider test that doesn't explicitly set `.usage` would otherwise leak a `MagicMock` into `Span.token_count` — which fails pydantic validation (`int | None`) and would break tests that never asked for tracing. `_extract_token_count` in each of the four judge provider files checks `isinstance(value, int)` and falls back to `None` rather than assuming the attribute access itself will fail.

**`HybridRetriever.retrieve` is not separately instrumented** — `Span` has no parent/child field, so `Trace.spans` is a flat list. Wrapping the coordinating method in addition to the four leaf calls it makes (`dense.retrieve`, `sparse.retrieve`, `reciprocal_rank_fusion`, `reranker.rerank`) would add a fifth, redundant `retrieval`-step span with no field to express "this one contains those."

**`Span.confidence_score` left unset by this task** — Populating it requires mapping the continuous 0–1 floats this codebase actually computes (retrieval similarity, `ConfidenceScore.composite`) onto the model's `1-5` int range. No such mapping is specified anywhere in the project docs; inventing one wasn't part of this task's scope, so it's left for whichever future orchestrator decides that conversion.

**`traced()` retyped with `ParamSpec`/`TypeVar` before the ninth call site, not left as `Callable[..., Any]` with a `# type: ignore` per site** — `traced()` was originally typed with `Callable[..., Any]`, which is what mypy strict mode considers an "untyped decorator": wrapping any function with it erases that function's real parameter and return types, and mypy flags the decoration itself as an error unless suppressed. This first surfaced instrumenting `DenseRetriever.retrieve` — the first real `@traced(...)` call site after the decorator itself was built — and eight more call sites across Tasks 6–15 were about to apply the same decorator. Accepting a `# type: ignore` at each one would have scattered nine (and growing) suppressions for the same root cause instead of one. `traced()` was retyped using `ParamSpec`/`TypeVar` (PEP 612) — `Callable[[Callable[P, T]], Callable[P, T]]` — so the wrapper preserves the wrapped function's exact signature, fixing the error at its source; every later call site (Tasks 6–15) confirms zero `# type: ignore` comments were needed anywhere in the codebase.

---

## 2026-07-04 — Trace/Span Data Models

**Plain (non-frozen) pydantic `BaseModel`, not the frozen-dataclass convention used for `FallbackResponse`/`ConfidenceScore`** — Every other judge-free result type in this codebase (`FallbackResponse`, `CitationVerificationResult`) is a frozen `@dataclass`, since none of them are ever handed to an LLM SDK or need to cross a process boundary. `Trace`/`Span` are different: they're the record a future JSON file writer and SQLite indexer will persist, so they need reliable JSON (de)serialization. `ProcessedDocument`/`Chunk` in `src/ingestion/models.py` face the identical requirement and are already plain pydantic models for the same reason — `Trace`/`Span` follow that precedent rather than the frozen-dataclass one.

**Closed `Literal`s for `step` and `status`, not open `str`** — Matches `ChunkingStrategy`'s existing convention (`src/ingestion/models.py`) for enumerations that are fixed by the project spec: the five pipeline steps (ingestion/retrieval/ranking/generation/verification) and three trace statuses (success/failure/degraded) are named explicitly in the spec, so a typo should fail validation immediately rather than silently producing an uncategorizable trace or span.

**`Span.confidence_score` is optional (`1-5` when present), not required** — Not every pipeline step naturally produces a confidence score — an ingestion span has no LLM-judged output to score, while a generation or verification span might. Requiring it on every span would force non-scoring steps to supply a meaningless placeholder value.

**`Trace.status` is required, with no default** — Unlike `Trace.spans` (defaults to `[]`) and `Trace.final_output` (defaults to `None`), `status` has no default. The object represents "the complete record of what happened" per the project spec's own phrasing — callers (the future context manager) must state the outcome explicitly once a request finishes, rather than the model silently assuming an optimistic default like `"success"`.

**Standalone models, not wired into a context manager, decorator, or writer** — Same situation as every Phase 2 generation module before an orchestrator existed: `Trace`/`Span` are usable and fully tested today, but nothing yet constructs a `Span` automatically as pipeline steps run, appends it to a `Trace`, or persists the finished `Trace` to JSON/SQLite. Those are separate, later tasks in the tracing module.

---

## 2026-07-04 — Graceful Fallback for Low Retrieval Confidence

**Trigger on `retrieval_confidence`, not the composite score** — The composite mixes in citation coverage and answer completeness, either of which can be low for reasons that have nothing to do with whether the right documents were retrieved (e.g. the LLM under-cited a well-grounded answer). The project spec calls out "retrieval confidence" by name for this decision, so `build_fallback_response` takes `ConfidenceScore.retrieval_confidence` as an explicit parameter rather than `ConfidenceScore.composite`.

**No LLM call** — Unlike answer completeness, this decision only needs the similarity scores and metadata already attached to `VectorStoreHit` (mean similarity vs. a threshold, plus title/section/source_path for the summary). Making it a judge call would add latency and cost to answer the same question a threshold comparison already answers deterministically.

**`FallbackResponse` is a frozen dataclass, not a pydantic `BaseModel`** — `JudgeVerdict`/`CompletenessVerdict` are pydantic models specifically because they're passed as `output_format=`/`response_format=` to LLM SDKs' structured-output APIs. `FallbackResponse` is never handed to an LLM, so it follows the codebase's default frozen-dataclass convention instead (matches `CitationVerificationResult`).

**`>=` threshold convention, matching the existing dedup check** — `retrieval_confidence >= threshold` means "confident enough, proceed with generation" (returns `None`). This mirrors `ChromaVectorStore`'s duplicate check (`(1.0 - distances[0]) >= self._threshold`) rather than inventing a new boundary convention for thresholds in this codebase.

**Fallback response is a standalone unit, not wired into a generation orchestrator** — Same situation as citation verification and confidence scoring: no code yet calls an LLM to produce the initial grounded answer, so callers are expected to compute a `ConfidenceScore` first and pass its `retrieval_confidence` in explicitly. It will be composed into an end-to-end `ask()` flow once that orchestrator exists.

---

## 2026-07-03 — Answer Confidence Scoring

**Boolean `complete`/`incomplete` verdict, mapped to `1.0`/`0.0`, rather than a continuous completeness score** — Matches the existing `JudgeVerdict.supported` boolean pattern from citation verification rather than asking the judge for a 1-5 or 0-1 score directly. A binary verdict is easier for an LLM judge to return consistently and easier to unit-test with canned fixtures than a continuous score whose exact numeric value would otherwise need its own calibration.

**`score_confidence` takes plain `float` weight parameters, not a `Settings` object** — Mirrors `reciprocal_rank_fusion(dense_weight=0.7, sparse_weight=0.3, ...)` in `src/retrieval/fusion.py`: the pure aggregation function stays decoupled from `Settings` and fully testable without constructing one; the future composition-root caller passes `settings.confidence_retrieval_weight` etc. explicitly, same as `HybridRetriever` already does for RRF's weights.

**Citation coverage is `0.0`, not `1.0`, when no citations were parsed** — An answer with zero verified citations has provided zero evidence of grounding, so it should score low on this dimension rather than scoring a vacuous "100% of nothing." This keeps `citation_coverage` interpretable as "how much of what was claimed is actually backed by evidence," not "how much of what little we checked passed."

**No new `pyproject.toml` extras or API keys** — The answer-completeness judge reuses `embed-openai` (`openai>=1.92.0`) and `llm-anthropic` (`anthropic>=0.100.0`), exactly as `CitationJudgeProtocol`'s providers already do. Both features share the same SDKs and the same `OPENAI_API_KEY`/`ANTHROPIC_API_KEY`, so there is no reason to add a parallel dependency surface.

**Confidence scoring is a standalone unit, not wired into a generation orchestrator** — Same situation as citation verification: no code yet calls an LLM to produce the initial grounded answer, so `score_confidence(query, answer_text, hits, citation_results, judge, ...)` takes all of these as plain parameters rather than generating any of them itself. It will be composed into an end-to-end `ask()` flow once that orchestrator exists.

**The "below-threshold → I don't know" fallback from the project spec is out of scope for this feature** — `score_confidence` returns the composite score and its three dimensions; deciding whether a low score should trigger `INSUFFICIENT_CONTEXT_RESPONSE` (already defined in `src/generation/prompts.py`) is left to the not-yet-built orchestrator, consistent with confidence scoring being a standalone unit.

---

## 2026-07-03 — Citation Verification

**`JudgeVerdict` is a pydantic `BaseModel`, breaking the codebase's usual frozen-dataclass convention for result types** — Every other pipeline result type (`VectorStoreHit`, `Citation`, `CitationVerificationResult`) is a frozen `@dataclass`. `JudgeVerdict` isn't, because both LLM SDKs' structured-output APIs (Anthropic's `messages.parse(output_format=...)`, OpenAI's `chat.completions.parse(response_format=...)`) require a pydantic model class as the schema argument, and the returned parsed object is itself an instance of exactly that class. Defining a separate dataclass and hand-converting on every provider call would add a translation step with no benefit; passing `JudgeVerdict` straight through keeps both providers' `judge()` implementations a single call plus a null-check.

**One judge call per citation, no batching** — `verify_citations` could in principle batch several citation-claim pairs into one LLM call to cut cost and latency. It doesn't, because the project spec's own wording ("send *each* citation-claim pair to an LLM-as-judge") specifies per-pair verification, and per-pair calls keep each verdict's evidence isolated — a single malformed or adversarial chunk can only ever influence its own citation's judgment, not bleed into a neighboring claim's evidence within the same prompt. Batching is a plausible future optimization if judge-call cost becomes a bottleneck, not a correctness requirement today.

**Out-of-range citation indices are rejected before calling the judge, not sent to the LLM to evaluate** — A citation like `[7]` when only 5 chunks were retrieved is unambiguously wrong — there is no chunk 7 to check. Sending it to the judge anyway would cost a real API call to answer a question the pipeline already knows the answer to (unsupported), and would require the judge prompt to handle a "no evidence available" case it doesn't otherwise need. Short-circuiting locally is both cheaper and enforces the grounded prompt's own "do not fabricate citations" instruction mechanically rather than trusting the judge to catch it.

**Citation verification is a standalone unit, not wired into a generation orchestrator** — The codebase has no code yet that actually calls an LLM to produce the initial grounded answer (`src/generation/prompts.py` only builds the prompt). Building that orchestrator was out of scope for this feature — `verify_citations(answer_text, hits, judge)` takes already-generated answer text as a plain parameter, so it's fully testable and usable today, and will be composed into an end-to-end `ask()` flow once that orchestrator exists, rather than this feature reaching backward to build it as a side effect.

**`RuntimeError`, not a bare `assert`, when an SDK returns no parsed structured output** — Both provider `judge()` implementations narrow the SDK's `Optional[ResponseFormatT]` return type before returning it. An `assert` was tried first but rejected in review: Python strips `assert` statements under `python -O`/`PYTHONOPTIMIZE=1`, which would let `None` propagate silently into the caller, later surfacing as an opaque `AttributeError` on `verdict.supported` instead of a clear, actionable error naming the actual failure.

**Nonce-tag wrapping (`wrap_with_nonce`) is shared between the grounded prompt and the judge prompt, not reimplemented** — Both prompts embed untrusted text (retrieved chunk content in the grounded prompt; claim and evidence text in the judge prompt, the latter also chunk-derived) that could contain an attempted prompt injection. Extracting the existing nonce-boundary logic from `build_grounded_prompt` into a standalone `wrap_with_nonce` helper means both call sites get the identical spotlighting defense with one implementation to audit, rather than two independently-written (and potentially divergent) copies.

**`openai>=1.92.0` version floor, bumped from the embedding extra's inherited `openai>=1.30`** — `OpenAICitationJudge` relies on `client.chat.completions.parse(..., response_format=...)`, the stable (non-beta) structured-output API. Version-bisecting PyPI wheels showed this method was only promoted out of `client.beta.chat.completions.parse` into the stable namespace in `openai==1.92.0` (`1.91.0` still lacks it). The repo's prior floor of `1.30` — inherited from the `embed-openai` extra, which only needs the embeddings endpoint — predates that promotion by many releases, so it would have silently allowed installing a version where this provider's `judge()` raises `AttributeError` at runtime. `pyproject.toml` now pins `openai>=1.92.0` for this feature's extra.

---

## 2026-07-03 — Cohere & Voyage Reranker Providers

**Equality check against the sentence_transformers default, not prefix matching, for reranker model-default resolution** — `make_embedder` distinguishes "the user left this on its default" from "the user chose a provider-specific model" by checking a model-name prefix (e.g. `text-embedding-*` for OpenAI). That trick fails for rerankers: Cohere's model (`rerank-v4.0-pro`) and Voyage's model (`rerank-2.5`) both start with `rerank-`, so a prefix check can't tell which provider a customized `RERANKER_MODEL` was meant for. Instead, `make_reranker` substitutes a provider's own default only when `settings.reranker_model` still exactly equals the sentence_transformers default — the one case unambiguously meaning "untouched" — and passes any other value through verbatim, so an explicit user choice is never silently overridden by a guess.

**Both new providers preserve the hosted API's returned order rather than re-sorting locally** — Cohere and Voyage's rerank endpoints already return results sorted descending by relevance score; both `CohereReranker.rerank()` and `VoyageReranker.rerank()` map results back onto `VectorStoreHit`s by iterating the API response in its returned order. Re-sorting locally would be redundant work that risks introducing a tie-breaking inconsistency with the provider's own ranking.

**No new API keys or `pyproject.toml` extras for reranking** — `cohere_api_key`/`voyage_api_key` and the `embed-cohere`/`embed-voyage` optional extras already exist from the embedding providers, and both rerank endpoints live in the same SDK packages. Adding parallel `rerank_cohere_api_key`-style settings or separate extras would duplicate configuration for no benefit — a user who wants Cohere for either embedding or reranking installs the same extra and sets the same key.

---

## 2026-07-02 — Cross-Encoder Reranker

**`sentence-transformers` CrossEncoder over LLM-as-judge for the default reranker** — The spec allows either "a small model or LLM-as-judge." A local cross-encoder (`cross-encoder/ms-marco-MiniLM-L6-v2`) requires no API key, no network round-trip, and no per-query cost, matching the project's existing default-to-local-model bias (`EMBEDDING_PROVIDER=sentence_transformers` is likewise the default embedder). LLM-as-judge reranking is left as a documented future provider: it would add per-query LLM cost and latency to every retrieval call and duplicates capability Phase 4's LLM-as-judge root-cause analysis will already exercise.

**`rerank_candidate_pool` (new) vs. `rerank_top_n` (existing, unchanged) — the RRF cutoff and the final answer size are now two separate settings** — Previously RRF's `top_n` was fed directly from `rerank_top_n` (5), so RRF itself performed the final cut and no reranking pass existed. Reusing `rerank_top_n` for the larger candidate pool would have silently changed its meaning for anyone already relying on "top 5" as the final answer size. Instead, `rerank_candidate_pool` (default 20) is a new, additive setting for RRF's output size; `rerank_top_n` keeps its original meaning — the number of chunks that reach generation — whether or not a reranker is installed. A `model_validator` enforces `rerank_top_n <= rerank_candidate_pool`.

**Reranker as an injected, optional dependency on `HybridRetriever`, gated by `reranking_enabled`** — Consistent with `Indexer`'s existing optional-constructor-arg pattern for `embedder`/`vector_store`/`bm25_store`. `HybridRetriever` never imports `make_reranker` itself; composition-root code decides whether to build and pass one. When no reranker is injected, or `reranking_enabled=False`, `retrieve()` falls back to slicing RRF's candidate pool directly — today's exact pre-reranker behavior — so existing callers and tests are unaffected without code changes.

**Cross-encoder scores reuse the `similarity` field on `VectorStoreHit`, via `dataclasses.replace()`** — Same convention as RRF (weighted rank score) and BM25 (max-normalized score) before it: each pipeline stage's current relevance signal overwrites `similarity` on a fresh frozen-dataclass copy rather than introducing a new field per stage. Cross-encoder scores are unbounded and model-dependent (not guaranteed to lie in [0, 1]), same caveat that already applies to raw RRF scores stored in this field today.

---

## 2026-07-02 — Required `embedder` on `ChromaVectorStore`

**`embedder` is a required constructor argument, not `Optional`** — `ChromaVectorStore` previously accepted `embedder=None`, which silently skipped both metadata stamping and the provider/dimension mismatch guard when the class was constructed directly (bypassing `make_vector_store`, which already required an embedder). A collection built this way could be reopened later under a mismatched embedding provider with no warning, surfacing only as a raw ChromaDB dimension error deep inside a query. Requiring `embedder` unconditionally closes that bypass and makes the two construction paths (`make_vector_store` and direct instantiation) enforce the same guarantee, matching the project's "fail fast at startup, not silently at query time" principle already used for the dimension guard itself.

---

## 2026-06-28 — Phase 1 Scaffold

**pyproject.toml as canonical dependency file** — Single source of truth for dependencies; `requirements.txt` is generated via `pip freeze` for locked reproducibility. Dev extras in `[project.optional-dependencies]` so `pip install -e ".[dev]"` installs everything in one step.

**Editable install (`pip install -e .`)** — `src/` is importable as `from src.config import settings` without reinstalling after edits. Keeps the test/run cycle fast.

**pydantic-settings for configuration** — All env vars are declared as typed fields with defaults and validators in `src/config.py`. The module-level `settings` singleton is the only import other modules need. Tests call `Settings()` directly (not the singleton) so `monkeypatch` env changes take effect per-test.

**Phase-scoped dependencies** — Only Phase 1 packages installed: `openai`, `chromadb`, `rank-bm25`, `langchain-text-splitters`, `langchain-community`, `pypdf`, `beautifulsoup4`, `numpy`, `sentence-transformers`, `pydantic-settings`. FastAPI, OpenTelemetry, Streamlit, and SQLite drivers are deferred to their respective phases to keep the environment lean.

**Stub `__init__.py` for future phases** — Modules for Phases 2-7 exist as docstring-only stubs. This makes the intended structure visible and importable without any implementation prematurely committed.

**`.gitkeep` for data directories** — `data/raw/`, `data/processed/`, `data/traces/`, `data/eval/`, `data/chroma/` are tracked in git via `.gitkeep` files. Runtime artifacts in those directories are excluded via `.gitignore` patterns.

---

## 2026-06-28 — Document Loader

**Simple dispatcher over a plugin registry** — `DocumentLoader.load()` inspects the file extension and calls a private format-specific function. Adding a new format requires one new function and one new branch in the dispatcher. A plugin registry would add indirection for no gain at four supported formats.

**SHA-256 of raw bytes as `doc_id`** — The ID is computed before any processing, so all documents produced from the same file share the same ID regardless of when they were loaded. This makes downstream deduplication a simple equality check on `doc_id` rather than a similarity scan.

**One `ProcessedDocument` per section (Markdown/HTML), per page (PDF), per file (plain text)** — The loader normalises structure, not chunks. Chunking is Phase 1's next step and operates on `ProcessedDocument` objects. Mixing loading and chunking in one pass would make both harder to test and replace independently.

**Section heading extraction is best-effort, not uniform** — Markdown and HTML have reliable structural markers (`#`/`<h1>`–`<h6>`); the loader extracts them. PDF has no reliable heading signals without layout analysis; the loader records `None` and uses the page boundary instead. Plain text has no structure at all. Honesty about what each format can provide is better than fabricating headings.

**Storage mirrors the source path** — Processed output lives at `data/processed/<filename>/page_NNN.json` or `section_NNN.json`. The directory name matches the source filename, making the link between raw and processed files immediately obvious without a database lookup. Re-ingesting overwrites the directory entirely; because `doc_id` is deterministic, downstream consumers detect unchanged content without re-embedding.

**Regex over `markdown-it-py` token stream for heading extraction** — The `markdown-it-py` token stream works well for full rendering but requires tracking parent tokens to identify headings. A single `re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)` over the raw text is simpler, equally correct for well-formed Markdown, and easier to reason about.

---

## 2026-06-28 — Chunking

**Three switchable strategies via `Settings.chunk_strategy`** — `fixed_size`, `recursive_header`, and `semantic` are controlled by a single env var. The dispatcher in `Chunker.chunk()` selects one strategy per call. All three preserve the full `ProcessedDocument` metadata on each `Chunk`, so downstream consumers don't need the source document.

**`chunk_id = sha256("{doc_id}:{text}")` (content-addressed)** — Identical text from the same document always produces the same `chunk_id` regardless of chunking settings. This makes deduplication a hash lookup rather than a similarity scan, matching the design of `doc_id` in the loader.

**Semantic strategy uses cosine distance, not similarity** — `distances = 1.0 - cosine_similarity`. Breakpoints are sentences where `distance > np.percentile(distances, semantic_breakpoint_percentile)`. Using distance (higher = more different) makes the threshold direction intuitive: "split at the top X% most dissimilar adjacent sentences."

**`chunk_overlap` is silently ignored in semantic mode** — Semantic chunks follow natural topic boundaries; overlap would duplicate sentence content across topic groups, which defeats the purpose. The `chunk_overlap` setting is validated globally (must be < `chunk_size`) but not applied in the semantic path. Callers that set `CHUNK_OVERLAP` with `CHUNK_STRATEGY=semantic` should expect no overlap.

**Lazy OpenAI client instantiation in `_semantic`** — The client is declared `None` and instantiated only when the first multi-sentence document is processed. Single-sentence documents skip the embeddings call and never create the client, so `CHUNK_STRATEGY=semantic` with only short documents incurs no API interaction.

**Cross-field validator: `chunk_overlap < chunk_size`** — `RecursiveCharacterTextSplitter` raises an opaque error when overlap ≥ size. A `model_validator(mode="after")` on `Settings` catches this at startup with a clear message rather than at chunking time.

---

## 2026-06-29 — Retrieval Indexing

**ChromaDB cosine space via `metadata={"hnsw:space": "cosine"}`** — ChromaDB defaults to L2 distance. Cosine similarity is the correct metric for normalized embedding vectors: two semantically identical sentences embedded separately should score 1.0 regardless of magnitude. Setting the space at collection creation time makes the deduplication formula `(1 - distance) >= threshold` straightforward and scale-invariant.

**`COLLECTION_NAME = "rag_chunks"` as an exported constant** — Other modules (tests, future retrieval query path) need to open the same collection by name. Exporting it from `vector_store.py` gives a single source of truth rather than a magic string repeated across files.

**Deduplication via batched nearest-neighbour query** — All candidate embeddings are passed to `collection.query()` in a single call (`n_results=1`); ChromaDB returns one result set per query embedding. `results["distances"][i][0]` gives the cosine distance from candidate i to its closest stored neighbour. A chunk is a duplicate when `(1 - distance) >= threshold`. This is O(1) round-trips regardless of batch size and is idiomatic to ChromaDB's multi-query API.

**`BM25Okapi` corpus uses `text.lower().split()` tokenization** — Consistent tokenization between index and query is more important than sophisticated tokenization for this domain. Lowercasing normalizes case variations; whitespace splitting is deterministic and dependency-free. The same `_tokenize` function is used for both `add` and `get_scores`, guaranteeing consistency.

**BM25 index persisted as pickled `{chunk_ids, corpus}`, not the `BM25Okapi` object** — Pickling the `BM25Okapi` object would couple the saved state to the installed version of `rank_bm25`. Pickling only the raw corpus and rebuilding on `load()` avoids deserialization errors when the library is upgraded. The rebuild cost is negligible for the corpus sizes expected in Phase 1.

**BM25 persistence path: `Path(settings.chroma_persist_dir).parent / "bm25_index.pkl"`** — Both indexes live under `data/`: ChromaDB at `data/chroma/` and BM25 at `data/bm25_index.pkl`. Co-locating them under the same parent makes backup and migration simple (copy `data/`), and the path is fully derived from the existing `CHROMA_PERSIST_DIR` setting without introducing a new env var.

**Parallel upsert via `ThreadPoolExecutor`, futures collected outside the `with` block** — ChromaDB upsert (I/O-bound) and BM25 `add` (CPU-bound, in-memory) are independent and benefit from concurrent execution. The `with ThreadPoolExecutor` block contains only `executor.submit()` calls; `.result()` collection happens after the context exits. This matters for exception safety: `__exit__` calls `shutdown(wait=True)`, ensuring both threads complete before any result is inspected or `bm25_store.save()` is called. If either future raises, the other thread has already finished and `save()` is skipped — leaving BM25 disk state consistent with ChromaDB on the next `Indexer()` construction.

**`BM25Okapi` requires ≥ 3 documents for discriminative IDF** — With exactly 2 documents, `IDF(term) = log((2 - 1 + 0.5) / (1 + 0.5)) = log(1) = 0` for any term appearing in exactly one document, making all scores 0.0. Tests that assert ranking behaviour use ≥ 3 documents to ensure non-zero IDF values. This is a property of the BM25 formula, not a library bug.

**Optional constructor args for `Indexer` (`embedder`, `vector_store`, `bm25_store`)** — The orchestrator creates its dependencies internally by default, but accepts pre-built instances via keyword-only args. This enables unit tests to inject a mock `Embedder` alongside real `VectorStore` and `BM25Store` instances backed by `tmp_path`, testing sync and dedup behaviour without calling the OpenAI API. The pattern is consistent with how `Chunker` tests mock `openai.OpenAI`.

---

## 2026-06-30 — RRF Fusion Layer

**RRF constant `k = 60` (Cormack et al. 2009)** — Standard value from the original RRF paper, used universally in production IR systems. Higher k flattens rank differences (top vs. bottom matter less); lower k exaggerates them. 60 is the empirically validated sweet spot for combining ranked lists in document retrieval.

**Weighted RRF, not simple RRF** — `score = weight / (k + rank)` per list, not `1 / (k + rank)`. Allows tuning the dense/sparse balance without changing retrieval k values. Default 0.7/0.3 reflects that dense retrieval generally outperforms BM25 on semantic questions while sparse still adds value for exact keyword matches (function names, config keys, error codes).

**Dense hit metadata priority on overlap** — When a chunk appears in both dense and sparse results, the dense hit's metadata is used. Dense retrieval fetches metadata from ChromaDB directly; sparse hydrates via a second ChromaDB lookup. Using dense metadata avoids a redundant round-trip on overlapping chunks and prefers the richer semantic retrieval path.

**`dataclasses.replace()` to set RRF score on frozen hits** — `VectorStoreHit` is a frozen dataclass. The fusion layer must update `similarity` to the RRF score without mutating the original hit. `dataclasses.replace(hit, similarity=rrf_score)` creates a new instance with all other fields copied — the idiomatic pattern for updating frozen dataclasses. Consistent with how `SparseRetriever` swaps in normalized BM25 scores.

**`HybridRetriever` as thin wiring layer, not algorithm** — `HybridRetriever` contains no retrieval or scoring logic; it reads `Settings`, calls both retrievers, and passes results to `reciprocal_rank_fusion`. Keeps the RRF algorithm independently testable as a pure function (11 unit tests, no mock retrievers) and makes `HybridRetriever` tests focus on wiring only (8 tests, both retrievers mocked).

---

## 2026-06-30 — Hybrid Retrieval Query Path

**`DenseRetriever` wraps `Embedder` + `VectorStore.query`** — The retriever is a thin stateless facade: embed query → query collection. Keeping embedding and querying separate (rather than adding a `query_text` method to `VectorStore`) means tests can swap `Embedder` for a fixture without touching `VectorStore`, and the same `Embedder` instance is shared across the indexing and query paths with no duplication.

**`SparseRetriever` fetches metadata from `VectorStore`, not `BM25Store`** — BM25 stores only `chunk_ids` and `corpus` (raw tokens) for index construction. Duplicating full chunk metadata (text, title, source_path, etc.) into the BM25 pickle would double storage and create a sync hazard. Instead, `SparseRetriever` calls `VectorStore.get_by_ids(ids)` to hydrate results — one ChromaDB round-trip per query regardless of `k`.

**BM25 scores normalized by max score before returning** — Raw BM25 scores are unbounded and dataset-dependent. Dividing by `max_score` maps results to (0, 1], making the `similarity` field semantically comparable to cosine similarity. This is required for RRF fusion (next step) and for displaying consistent confidence scores across retrieval modes.

**Zero-score guard in `SparseRetriever`** — After sorting, chunks with `score == 0.0` are dropped before normalization. A zero-score chunk means BM25 found no query term overlap — including it would produce `0 / max_score = 0` similarity entries that pollute fusion results. The guard also prevents division-by-zero when all scores are zero (empty index or no term overlap).

**`VectorStoreHit` as the shared result model** — Both `DenseRetriever` and `SparseRetriever` return `list[VectorStoreHit]`. Using a single frozen dataclass means the RRF fusion layer and reranker don't need to know which retriever produced a hit. `dataclasses.replace` in `SparseRetriever` swaps in the normalized BM25 score without mutating the ChromaDB-returned hit.

---

## 2026-06-30 — Embedding & Vector Store Provider Abstraction

**`Protocol` + factory over inheritance** — `EmbedderProtocol` and `VectorStoreProtocol` are structural (`typing.Protocol`), not abstract base classes. Any object with the right method signatures satisfies the interface without inheriting from anything, which keeps provider modules free of coupling to a shared base class and lets `runtime_checkable` isinstance checks work in tests.

**`sentence_transformers` as the default embedding provider, not `openai`** — Running the pipeline out of the box (`cp .env.example .env && pytest`) should not require an API key. `text-embedding-3-small` (OpenAI, 1536 dims) remains available for production use, but `all-MiniLM-L6-v2` (sentence-transformers, 384 dims, already a base dependency) is what a fresh clone uses by default.

**Collection metadata as the dimension guard, not a config-only check** — Embedding dimensions differ by provider and are fixed on a ChromaDB collection after the first insert. Rather than trust the running config alone, `ChromaVectorStore` writes `embedding_provider`/`embedding_dimensions` into the collection metadata on first creation and compares against it on every later open, raising a clear `ValueError` naming both the stored and configured provider if they diverge. This turns a silent dimension-mismatch corruption into a startup-time error.

**Lazy provider SDK imports inside factory branches, not at module level** — `make_embedder` imports `src.retrieval.providers.embedder_openai` or `embedder_sentence_transformers` only inside the matching `if provider == ...` branch. This means `pip install -e ".[dev]"` (no provider extras) doesn't fail at import time just because `openai` isn't installed, and installing `embed-openai` doesn't force `sentence-transformers` model downloads for users who never select that provider.

**`Embedder`/`VectorStore` kept as backward-compatible aliases via module `__getattr__`** — Existing call sites (and pre-refactor tests) that import the old concrete class names keep working. `__getattr__` resolves them lazily to `OpenAIEmbedder`/`ChromaVectorStore` on first access instead of importing provider SDKs eagerly at module load.

**Model-name guard in `make_embedder`** — If `settings.embedding_provider == "sentence_transformers"` but `settings.embedding_model` still holds an OpenAI-style name (prefix `text-embedding`, the field's own default), the factory substitutes the sentence-transformers default model instead of attempting to load an OpenAI model name locally. Prevents a confusing `sentence-transformers` load error when a user changes only `EMBEDDING_PROVIDER` and forgets `EMBEDDING_MODEL`.
