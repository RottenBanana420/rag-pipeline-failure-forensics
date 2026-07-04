# Architecture Overview

## 2026-07-04 — Phase 3: Trace/Span Data Models (Complete)

### The Record of What Happened

`Span` and `Trace` (`src/tracing/models.py`) are the data model a future context manager/decorator will populate as pipeline steps run, and a future JSON/SQLite writer will persist. This entry covers only the models — no instrumentation exists yet to construct them automatically.

**`Span`** — one pipeline step's record:

| Field | Type | Description |
|-------|------|--------------|
| `span_id` | `str` | Auto-generated UUID4 hex, unique per instance |
| `step` | `Literal["ingestion", "retrieval", "ranking", "generation", "verification"]` | Which pipeline stage produced this span |
| `input` | `str` | Serialized step input — serialization itself is the future decorator's job |
| `output` | `str` | Serialized step output |
| `llm_prompt` | `str \| None` | The prompt sent, if this step called an LLM |
| `token_count` | `int \| None` | `>= 0`; `None` for non-LLM steps |
| `latency_ms` | `float` | `>= 0.0` |
| `confidence_score` | `int \| None` | `1-5`; optional because not every step naturally produces one (e.g. ingestion) |

**`Trace`** — the complete record for one request:

| Field | Type | Description |
|-------|------|--------------|
| `trace_id` | `str` | Auto-generated UUID4 hex, unique per instance |
| `spans` | `list[Span]` | Defaults to `[]`; a future context manager appends as steps complete |
| `final_output` | `str \| None` | Defaults to `None` (e.g. a failed request may never produce output) |
| `status` | `Literal["success", "failure", "degraded"]` | Required, no default — callers must state the outcome explicitly |

**Public API:**

```python
from src.tracing.models import Span, Trace

span = Span(step="retrieval", input='{"question": "..."}', output='{"hits": [...]}', latency_ms=12.5)
trace = Trace(spans=[span], final_output="The answer is 42.", status="success")

# Round-trips for the future JSON/SQLite writers:
restored = Trace.model_validate_json(trace.model_dump_json())
```

**Design notes:**
- Both are plain (non-frozen) pydantic `BaseModel`s, matching `ProcessedDocument`/`Chunk` in `src/ingestion/models.py` — not the frozen-dataclass convention used for judge-free result values (`FallbackResponse`, `ConfidenceScore`) — because they need free JSON serialization for the not-yet-built writers, the same rationale `ProcessedDocument`/`Chunk` already established.
- `step` and `status` are closed `Literal`s, matching `ChunkingStrategy`'s convention, since both sets are fixed by the project spec.
- Standalone models only — the context manager, decorator, and JSON/SQLite writers described in the project spec are future work, same situation as every Phase 2 generation module before an orchestrator exists.

---

## 2026-07-04 — Phase 2: Graceful Fallback for Low Retrieval Confidence (Complete)

### Structured "Insufficient Information" Response

`build_fallback_response` closes the last gap called out in the confidence-scoring entry below: deciding what to do with a low retrieval-confidence score. It checks `ConfidenceScore.retrieval_confidence` specifically — not the composite — against a new `settings.retrieval_confidence_threshold` (default `0.5`). The composite mixes in citation coverage and answer completeness, which can be low for reasons unrelated to whether the right documents were found; the project spec calls out retrieval confidence by name for this decision.

Below threshold, it returns a frozen `FallbackResponse` instead of `None`:

- `message` — fixed framing text (`FALLBACK_MESSAGE`), parallel to `INSUFFICIENT_CONTEXT_RESPONSE` in `src/generation/prompts.py`
- `retrieved_summary` — one line per retrieved hit (title, section heading, similarity), or an explicit "nothing retrieved" line if `hits` is empty
- `documents_to_check` — deduplicated document identifiers ("worth checking manually"), ordered by descending similarity; hits are identified by `title`, falling back to `"title (source_path)"` only when two hits share a title but different source paths

No LLM call is involved — unlike citation coverage and answer completeness, this dimension only needs arithmetic over data already attached to `VectorStoreHit`, so it's deterministic and free to compute.

**Public API:**

```python
from src.config import settings
from src.generation import build_fallback_response, score_confidence

score = score_confidence(query, answer_text, hits, citation_results, judge, ...)
fallback = build_fallback_response(
    hits, score.retrieval_confidence, settings.retrieval_confidence_threshold
)
if fallback is not None:
    return fallback  # instead of the generated answer
```

**Design notes:**
- `retrieval_confidence >= threshold` → `None` (proceed with generation); this matches the `>=` convention `ChromaVectorStore` already uses for its dedup check (`(1.0 - distances[0]) >= self._threshold`) rather than introducing a new comparison convention.
- `FallbackResponse` is a frozen dataclass, not a pydantic `BaseModel` — there's no LLM structured-output call here to justify the pydantic convention used by `JudgeVerdict`/`CompletenessVerdict`.
- Like citation verification and confidence scoring, this is a standalone, directly-callable unit — the codebase has no generation orchestrator yet to call it automatically after generation.

---

## 2026-07-03 — Phase 2: Answer Confidence Scoring (Complete)

### Composite Confidence Score

`score_confidence` rates a generated answer on three independent dimensions and combines them into one composite score, closing the loop described in the project's "Confidence Scoring" spec.

**Retrieval confidence** — mean `similarity` across the `VectorStoreHit`s actually used for generation (`0.0` if none retrieved). All reranker providers already overwrite `similarity` with their own relevance score before this runs, so the signal is consistent regardless of which reranker (or none) produced the hits.

**Citation coverage** — the fraction of `verify_citations`' `CitationVerificationResult`s with `supported=True` (`0.0` if no citations were found). Pure arithmetic over Phase 2's existing citation verification output — no new LLM call.

**Answer completeness** — the one dimension that needs judgment: whether the answer addresses every part of the question. `CompletenessJudgeProtocol` (`judge(question, answer) -> CompletenessVerdict`, `provider_id`) mirrors `CitationJudgeProtocol` exactly. `make_completeness_judge(settings)` reads `settings.answer_completeness_judge_provider` and returns the matching implementation, same lazy-import factory pattern as `make_citation_judge`.

**Implemented providers:**

| Provider | Class | File | Default model |
|---|---|---|---|
| `anthropic` (default) | `AnthropicCompletenessJudge` | `src/generation/providers/completeness_judge_anthropic.py` | `claude-sonnet-4-5` |
| `openai` | `OpenAICompletenessJudge` | `src/generation/providers/completeness_judge_openai.py` | `gpt-4o-2024-08-06` |

**Public API:**

```python
from src.config import settings
from src.generation import make_completeness_judge, score_confidence

judge = make_completeness_judge(settings)
result = score_confidence(
    query, answer_text, hits, citation_results, judge,
    retrieval_weight=settings.confidence_retrieval_weight,
    citation_weight=settings.confidence_citation_weight,
    completeness_weight=settings.confidence_completeness_weight,
)
print(result.retrieval_confidence, result.citation_coverage, result.answer_completeness, result.composite)
```

**Design notes:**
- `CompletenessVerdict` is a pydantic `BaseModel` (`complete: bool`, `reasoning: str`), not a frozen dataclass — same rationale as `JudgeVerdict`: it's passed directly as the structured-output schema type to both providers' SDKs.
- `score_confidence` takes plain `float` weight parameters with defaults (not a `Settings` object), mirroring `reciprocal_rank_fusion(dense_weight=..., sparse_weight=...)` — the composite is an unnormalized weighted sum, same convention as RRF.
- This module is a standalone, directly-callable unit — like citation verification, there is no generation orchestrator yet to wire it into automatically. The "if retrieval confidence is below threshold, return a structured 'I don't know' response" fallback described in the project spec is explicitly out of scope here; `score_confidence` only returns the score.

---

## 2026-07-03 — Phase 2: Citation Verification (Complete)

### LLM-as-Judge Citation Checking

The grounded prompt (below) asks the generation LLM to cite every claim with `[N]` markers, but nothing verified those citations were honest until now. `verify_citations` closes that gap: it re-reads the model's own answer text, pairs each citation with the chunk(s) it claims to cite, and asks a second LLM call whether the cited text actually supports the claim.

**`parse_citations`** (`src/generation/citation_parser.py`): a v1 regex heuristic, not sentence-boundary NLP. It finds each contiguous run of `[N]` markers (`r"(?:\[\d+\])+"`) and pairs it with the text since the previous run (or start of string) as the claim. Good enough to bound "what text does this citation apply to" without a full parser.

**`CitationJudgeProtocol`** (`src/generation/citation_verifier.py`): `judge(claim, evidence) -> JudgeVerdict`, `provider_id: str`. `JudgeVerdict` is a pydantic `BaseModel` (`supported: bool`, `reasoning: str`) rather than the codebase's usual frozen dataclass — deliberately, so it can be passed straight through as the structured-output schema type to both providers' SDKs (`output_format=JudgeVerdict` for Anthropic, `response_format=JudgeVerdict` for OpenAI). `make_citation_judge(settings)` reads `settings.citation_judge_provider` and returns the matching implementation, same lazy-import factory pattern as `make_embedder`/`make_reranker`.

**Implemented providers:**

| Provider | Class | File | Default model |
|---|---|---|---|
| `anthropic` (default) | `AnthropicCitationJudge` | `src/generation/providers/citation_judge_anthropic.py` | `claude-sonnet-4-5` |
| `openai` | `OpenAICitationJudge` | `src/generation/providers/citation_judge_openai.py` | `gpt-4o-2024-08-06` |

**`verify_citations(answer_text, hits, judge) -> list[CitationVerificationResult]`:** resolves each citation's (1-indexed) chunk indices against `hits`. An index outside `1..len(hits)` is untrusted LLM output — the model referenced a chunk that doesn't exist — and short-circuits to an unsupported result without ever calling the judge. In-range citations get exactly one `judge.judge(claim, evidence)` call each (no batching), with the cited hits' text joined as evidence.

**Public API:**

```python
from src.config import settings
from src.generation import make_citation_judge, verify_citations

judge = make_citation_judge(settings)          # provider chosen by settings.citation_judge_provider
results = verify_citations(answer_text, hits, judge)
for r in results:
    print(r.supported, r.chunk_indices, r.reasoning)
```

**Design notes:**
- Prompt-injection defense reuses the grounded prompt's spotlighting pattern: `build_judge_prompt` wraps both the claim and the evidence in nonce-suffixed `<claim-...>`/`<evidence-...>` tags via `wrap_with_nonce` (extracted from `build_grounded_prompt` for this reuse), so neither the model's own citation text nor untrusted chunk content can forge a closing tag and escape its block.
- Both provider `judge()` implementations raise `RuntimeError` (not a bare `assert`) when the SDK returns no parsed structured output — `assert` is stripped under `python -O`, which would otherwise let a `None` verdict propagate into an opaque `AttributeError` several frames later.
- This module is a standalone, directly-callable unit — the codebase has no generation orchestrator yet (nothing calls an LLM to produce the initial grounded answer), so `verify_citations` currently takes `answer_text` as a plain parameter rather than generating it itself. Wiring it into an end-to-end `ask()` flow is future work.

---

## 2026-07-03 — Phase 1: Cohere & Voyage Reranker Providers (Complete)

### Additional Reranker Providers

Extends the reranker beyond the local `sentence_transformers` cross-encoder to two hosted rerank APIs, mirroring the embedder's multi-provider pattern.

| Provider | Class | File | Default model |
|---|---|---|---|
| `cohere` | `CohereReranker` | `src/retrieval/providers/reranker_cohere.py` | `rerank-v4.0-pro` |
| `voyage` | `VoyageReranker` | `src/retrieval/providers/reranker_voyage.py` | `rerank-2.5` |

Both follow `reranker_sentence_transformers.py`'s contract exactly (`rerank(query, hits, top_n) -> list[VectorStoreHit]`, results mapped back via `dataclasses.replace(hit, similarity=...)`, empty `hits` short-circuits without an API call) and reuse the embedder's existing `cohere_api_key`/`voyage_api_key` settings and `embed-cohere`/`embed-voyage` extras — no new API keys or extras needed.

**Model-default resolution can't reuse `make_embedder`'s prefix trick:** Cohere's model (`rerank-v4.0-pro`) and Voyage's model (`rerank-2.5`) both start with `"rerank-"`, so `make_reranker` can't tell them apart by prefix the way `make_embedder` distinguishes `text-embedding-*` from `voyage-*`. Instead, it applies a provider's own default only when `settings.reranker_model` still equals the sentence_transformers default (`cross-encoder/ms-marco-MiniLM-L6-v2`) — i.e., the user hasn't customized it at all — and otherwise passes the configured value through verbatim, trusting the user set it correctly for whichever provider they chose.

**Public API:**

```python
from src.config import Settings
from src.retrieval.reranker import make_reranker

settings = Settings(reranker_provider="cohere")   # or "voyage"
reranker = make_reranker(settings)
hits = reranker.rerank(query, candidate_hits, top_n=5)
```

**Design notes:**
- Both providers preserve the hosted API's returned order (already sorted descending by relevance) rather than re-sorting locally.
- Voyage's SDK parameter is `top_k`, not `top_n` like Cohere's — an easy name to mis-copy between the two nearly-identical provider modules; each was verified independently against current SDK docs before implementation.

---

## 2026-07-02 — Phase 1: Cross-Encoder Reranker (Complete)

### Second-Pass Reranking

RRF fusion ranks purely by rank position across the dense/sparse lists — it has no view of the actual query text once fusion runs. A reranker adds a precision pass: RRF now widens its own cutoff to a `rerank_candidate_pool` (default 20), and each candidate is re-scored against the literal query text before the final `rerank_top_n` (default 5) is kept.

**`RerankerProtocol`** (`src/retrieval/reranker.py`): `rerank(query, hits, top_n) -> list[VectorStoreHit]`, `provider_id: str`. `make_reranker(settings)` reads `settings.reranker_provider` and returns the matching implementation, lazily importing the provider SDK inside the factory branch — same pattern as `make_embedder`.

**Implemented provider:**

| Provider | Class | File | Default model |
|---|---|---|---|
| `sentence_transformers` (default) | `SentenceTransformersReranker` | `src/retrieval/providers/reranker_sentence_transformers.py` | `cross-encoder/ms-marco-MiniLM-L6-v2` |

LLM-as-judge reranking is documented as a future provider only — not implemented, to avoid duplicating capability Phase 4's LLM-as-judge root-cause analysis will already exercise.

**Settings split:** `rerank_candidate_pool` (new, default 20) is RRF's own cutoff; `rerank_top_n` (existing, default 5, unchanged meaning) is the final number of chunks reaching generation, whether or not a reranker is installed. A `model_validator` enforces `rerank_top_n <= rerank_candidate_pool`.

**`HybridRetriever`** accepts an optional `reranker: RerankerProtocol | None` constructor arg (mirrors `Indexer`'s optional `embedder`/`vector_store`/`bm25_store` pattern). When `reranking_enabled=False` or no reranker was injected, `retrieve()` falls back to slicing the RRF candidate pool directly — today's exact pre-reranker behavior, so no existing caller breaks.

**Public API:**

```python
from src.config import Settings
from src.retrieval.reranker import make_reranker
from src.retrieval.hybrid_retriever import HybridRetriever

settings = Settings()
reranker = make_reranker(settings)          # provider chosen by settings.reranker_provider

retriever = HybridRetriever(dense, sparse, settings, reranker=reranker)
hits = retriever.retrieve("how do I configure chunking?")  # list[VectorStoreHit], len <= rerank_top_n
```

**Design notes:**
- Cross-encoder scores overwrite `VectorStoreHit.similarity` via `dataclasses.replace()` — same convention RRF and BM25 already use at each pipeline stage. Scores are unbounded and model-dependent, not guaranteed to lie in [0, 1].
- Verified live against the real `cross-encoder/ms-marco-MiniLM-L6-v2` model (not just mocked unit tests): a lexically-favored but semantically irrelevant chunk correctly dropped from rank 1 to last after reranking, while the actually-relevant chunk rose from last to first.

---

## 2026-07-02 — Phase 1: Voyage, Gemini, Cohere Embedding Providers (Complete)

### Additional Embedding Providers

Fills in the three providers that were stubbed as "planned" in `Settings`'s `Literal` type and the factory's `NotImplementedError`/`ValueError` branches.

| Provider | Class | File | Default model | Batch size |
|---|---|---|---|---|
| `voyage` | `VoyageEmbedder` | `providers/embedder_voyage.py` | `voyage-3.5` | 128 (Voyage's documented rate-limit-safe batch) |
| `gemini` | `GeminiEmbedder` | `providers/embedder_gemini.py` | `gemini-embedding-001` | 100 (conservative default — no documented fixed limit) |
| `cohere` | `CohereEmbedder` | `providers/embedder_cohere.py` | `embed-v4.0` | 96 (Cohere's documented max per `embed()` call) |

**Gemini targets `google-genai`, not `google-generativeai`** — the unified SDK Google is standardizing on, superseding the legacy package originally pinned as a placeholder in `pyproject.toml`.

Each provider follows the pattern established by `OpenAIEmbedder`: batched calls sized to the provider's own documented limits, an API-key field on `Settings`, and a `make_embedder` branch with the same OpenAI-model-name guard (falls back to the provider's own default if `settings.embedding_model` looks like an OpenAI model name).

---

## 2026-07-02 — ChromaVectorStore: Required Embedder Guard

`ChromaVectorStore` previously accepted `embedder=None` and silently skipped both metadata stamping and the provider/dimension mismatch check when constructed directly — bypassing the guard that `make_vector_store` already enforced. A collection created this way could later be reopened under a mismatched embedding provider with no warning, until a raw ChromaDB dimension error surfaced deep inside a query.

`embedder` is now a required constructor argument, matching `make_vector_store`'s existing contract. Direct construction without an embedder now fails fast at startup instead of corrupting silently at query time.

---

## 2026-06-30 — Phase 1: Embedding & Vector Store Provider Abstraction (Complete)

### Provider Abstraction

`Embedder` and `VectorStore` were refactored from concrete OpenAI/ChromaDB classes into `Protocol`-based interfaces with factory functions, so switching providers is an environment variable change, not a code change.

**`EmbedderProtocol`** (`src/retrieval/embedder.py`): `embed(texts) -> list[list[float]]`, `dimensions: int`, `provider_id: str`. `make_embedder(settings)` reads `settings.embedding_provider` and returns the matching implementation, importing each provider SDK lazily inside the factory branch so installing one provider's optional extra doesn't pull in the others.

**`VectorStoreProtocol`** (`src/retrieval/vector_store.py`): `filter_duplicates`, `upsert`, `query`, `get_by_ids`, `count`. `make_vector_store(settings, embedder)` returns the configured implementation.

**Implemented providers:**

| Provider | Class | File | Requires |
|---|---|---|---|
| `sentence_transformers` (default) | `SentenceTransformersEmbedder` | `src/retrieval/providers/embedder_sentence_transformers.py` | Nothing — already a base dependency |
| `openai` | `OpenAIEmbedder` | `src/retrieval/providers/embedder_openai.py` | `OPENAI_API_KEY`, `pip install -e ".[embed-openai]"` |
| `voyage` | `VoyageEmbedder` | `src/retrieval/providers/embedder_voyage.py` | `VOYAGE_API_KEY`, `pip install -e ".[embed-voyage]"` |
| `gemini` | `GeminiEmbedder` | `src/retrieval/providers/embedder_gemini.py` | `GEMINI_API_KEY`, `pip install -e ".[embed-gemini]"` |
| `cohere` | `CohereEmbedder` | `src/retrieval/providers/embedder_cohere.py` | `COHERE_API_KEY`, `pip install -e ".[embed-cohere]"` |
| `chroma` (default vector store) | `ChromaVectorStore` | `src/retrieval/vector_store.py` | Nothing — already a base dependency |

The `qdrant` vector store is declared in `Settings`'s `Literal` type and `pyproject.toml`'s `store-qdrant` extra but not yet implemented; selecting it raises `NotImplementedError` from `make_vector_store`. All five embedding providers above are fully implemented.

**Dimension guard:** Embedding dimensions vary by provider (e.g. OpenAI `text-embedding-3-small` = 1536, `all-MiniLM-L6-v2` = 384). `ChromaVectorStore` stamps `embedding_provider` and `embedding_dimensions` into the collection's metadata the first time it's created. On every later open, if an `embedder` is passed and its `provider_id` doesn't match the stored metadata, construction raises `ValueError` with a message telling the user to delete `data/chroma/` and re-index — this prevents silently querying a collection with vectors from a different embedding space.

**`embedder_openai.py` model-name guard:** `make_embedder` won't blindly pass `settings.embedding_model` to `SentenceTransformersEmbedder` — if the configured model name starts with `text-embedding` (an OpenAI-style name) and the provider is `sentence_transformers`, it falls back to `SentenceTransformersEmbedder`'s own default (`all-MiniLM-L6-v2`) instead of trying to load an OpenAI model name as a local model, which would fail confusingly.

**Backward-compatible aliases:** `Embedder` and `VectorStore` remain importable (as aliases for `OpenAIEmbedder`/`ChromaVectorStore` respectively) via module-level `__getattr__`, so existing call sites and tests that import the old names keep working without eagerly importing every provider SDK at module load time.

**Public API:**

```python
from src.config import settings
from src.retrieval.embedder import make_embedder
from src.retrieval.vector_store import make_vector_store

embedder = make_embedder(settings)          # provider chosen by settings.embedding_provider
vector_store = make_vector_store(settings, embedder)  # provider chosen by settings.vector_store_provider
```

---

## 2026-06-30 — Phase 1: RRF Fusion + HybridRetriever (Complete)

### Fusion Layer

**`reciprocal_rank_fusion`** (`fusion.py`):

- Combines `dense_hits` and `sparse_hits` into a single ranked list
- Score per chunk: `sum_r: weight_r / (k + rank_r)` where `k = 60` (Cormack et al. 2009)
- Default weights: `dense_weight=0.7`, `sparse_weight=0.3` (configurable via `Settings`)
- When a chunk appears in both lists, scores accumulate — overlap boosts rank
- Dense hit metadata takes priority when a chunk appears in both lists (`hits_by_id[id] = hit` for dense, `setdefault` for sparse)
- Output: `list[VectorStoreHit]` with `similarity` set to RRF score (not original cosine/BM25 score)
- `top_n` limits final output (default: `settings.rerank_top_n = 5`)

**`HybridRetriever`** (`hybrid_retriever.py`):

- Wires `DenseRetriever` + `SparseRetriever` + `reciprocal_rank_fusion` via `Settings`
- `.retrieve(query)` → calls both retrievers with configured `k` → fuses → returns top-N

**Public API:**

```python
from src.retrieval import HybridRetriever, DenseRetriever, SparseRetriever, BM25Store
from src.retrieval.embedder import make_embedder
from src.retrieval.vector_store import make_vector_store
from src.config import Settings

settings = Settings()
embedder = make_embedder(settings)
vector_store = make_vector_store(settings, embedder)
dense = DenseRetriever(embedder, vector_store)
sparse = SparseRetriever(BM25Store(settings), vector_store)

retriever = HybridRetriever(dense, sparse, settings)
hits = retriever.retrieve("how do I configure chunking?")  # list[VectorStoreHit], len ≤ rerank_top_n
```

**Design notes:**
- `reciprocal_rank_fusion` is a pure function with no dependencies on retriever internals — independently testable (11 unit tests, no mocks)
- `HybridRetriever` contains no scoring logic; tests mock both retrievers and verify wiring only (8 tests)
- RRF scores replace `similarity` via `dataclasses.replace` — frozen `VectorStoreHit` instances are never mutated

---

## 2026-06-30 — Phase 1: Hybrid Retrieval — Dense & Sparse Retrievers (Complete)

### Retrieval Query Path

Two retriever classes implement the query side of hybrid retrieval. Both return `list[VectorStoreHit]`.

**`VectorStoreHit`** — shared result model for all retrieval paths:

| Field | Type | Description |
|-------|------|-------------|
| `chunk_id` | `str` | Content-addressed chunk ID |
| `text` | `str` | Chunk text |
| `doc_id` | `str` | Source document ID |
| `source_path` | `str` | Path to original file |
| `title` | `str` | Document title |
| `section_heading` | `str \| None` | Nearest section heading |
| `chunk_index` | `int` | 0-based position within document |
| `strategy` | `str` | Chunking strategy used |
| `similarity` | `float` | Relevance score in [0, 1] |

**`DenseRetriever`** (`dense_retriever.py`):
- Embeds the query via `Embedder.embed([query])` (single-item list, unpacked)
- Calls `VectorStore.query(embedding, k)` → ranked by cosine similarity
- Default `k=10`

**`SparseRetriever`** (`sparse_retriever.py`):
- Scores all indexed chunks via `BM25Store.get_scores(query)`
- Sorts by BM25 score descending, takes top-k, discards zero-score chunks
- Normalizes scores by dividing by max score → `similarity` in (0, 1]
- Fetches full chunk metadata from `VectorStore.get_by_ids(ids)` (single round-trip)
- Replaces `similarity` field with the normalized BM25 score via `dataclasses.replace`

**Public API:**

```python
from src.retrieval import DenseRetriever, SparseRetriever, BM25Store
from src.retrieval.embedder import make_embedder
from src.retrieval.vector_store import make_vector_store
from src.config import Settings

settings = Settings()
embedder = make_embedder(settings)
vector_store = make_vector_store(settings, embedder)
bm25_store = BM25Store(settings)

dense = DenseRetriever(embedder, vector_store)
sparse = SparseRetriever(bm25_store, vector_store)

dense_hits = dense.retrieve("how do I configure chunking?", k=10)
sparse_hits = sparse.retrieve("how do I configure chunking?", k=10)
```

**Design notes:**
- `SparseRetriever` depends on `VectorStore` to fetch metadata (text, source_path, title, etc.) — BM25 only stores chunk IDs and corpus tokens, not full metadata. This keeps BM25 persistence lightweight and avoids duplicating metadata.
- Score normalization (`/ max_score`) makes BM25 scores comparable to cosine similarity scores, which is required for RRF fusion in the next step.
- If BM25 has no scores or all scores are 0.0, `SparseRetriever` returns `[]` without touching ChromaDB.

---

## 2026-06-29 — Phase 1: Retrieval Indexing (Complete)

### Retrieval Module

The retrieval module (`src/retrieval/`) handles embedding, vector storage, BM25 indexing, and orchestrated ingestion into both indexes.

**Public API:**

```python
from src.retrieval import BM25Store, Indexer
from src.config import Settings

settings = Settings()
indexer = Indexer(settings)          # builds its own embedder/vector store via the factories, loads existing BM25 index from disk on init

# Full pipeline: embed → dedup → parallel upsert → save BM25
stored_ids = indexer.index(chunks)   # chunks: list[Chunk] from src.ingestion
```

**Classes:**

| Class | File | Responsibility |
|-------|------|----------------|
| `EmbedderProtocol` / `make_embedder` | `embedder.py` | Provider-agnostic embedding interface + factory (see provider abstraction entry above) |
| `OpenAIEmbedder` | `providers/embedder_openai.py` | Batch OpenAI embeddings (`BATCH_SIZE = 200`) |
| `SentenceTransformersEmbedder` | `providers/embedder_sentence_transformers.py` | Local embeddings via `sentence-transformers`, no API key |
| `VectorStoreProtocol` / `make_vector_store` | `vector_store.py` | Provider-agnostic vector store interface + factory |
| `ChromaVectorStore` | `vector_store.py` | ChromaDB persistent client, cosine-space collection `"rag_chunks"`, upsert + dedup + dimension guard |
| `BM25Store` | `bm25_store.py` | BM25Okapi index, cumulative add, pickle persistence, scored retrieval |
| `Indexer` | `indexer.py` | Orchestrator: embed → dedup → parallel upsert (ThreadPoolExecutor) → save BM25 |

**Deduplication:** `VectorStore.filter_duplicates` queries ChromaDB for the nearest neighbour of each incoming embedding. Chunks where `(1 - cosine_distance) >= settings.dedup_threshold` (default 0.95) are excluded from both stores. Fast-paths to accept all when the collection is empty.

**Sync guarantee:** `Indexer.index` applies deduplication before touching either store, so ChromaDB and BM25 always receive the same accepted set. The `ThreadPoolExecutor` `with` block is exited (which calls `shutdown(wait=True)`) before either `.result()` is collected — ensuring both threads complete before any exception propagates or `bm25_store.save()` is called.

**BM25 persistence:** `BM25Store` pickles `{"chunk_ids": list[str], "corpus": list[list[str]]}` to `data/bm25_index.pkl` (sibling of `data/chroma/`). On load it reconstructs `BM25Okapi` from the corpus. The `BM25Okapi` object itself is not pickled to avoid library-version compatibility issues.

**ChromaDB metadata** stored per chunk:

| Field | Type | Source |
|-------|------|--------|
| `source_path` | `str` | `chunk.source_path` |
| `chunk_index` | `int` | `chunk.chunk_index` |
| `section_heading` | `str` | `chunk.section_heading or ""` |
| `strategy` | `str` | `chunk.strategy` |
| `char_count` | `int` | `len(chunk.text)` |
| `doc_id` | `str` | `chunk.doc_id` |
| `title` | `str` | `chunk.title` |

---

## 2026-06-28 — Phase 1: Chunking (Complete)

### Chunking Module

The chunker (`src/ingestion/chunker.py`) splits `ProcessedDocument` objects into `Chunk` objects using one of three switchable strategies. Strategy is set via `Settings.chunk_strategy`.

**`Chunk`** — the output of chunking:

| Field | Type | Description |
|-------|------|-------------|
| `chunk_id` | `str` | SHA-256 of `"{doc_id}:{text}"` — deterministic content address |
| `doc_id` | `str` | Inherited from source `ProcessedDocument` |
| `source_path` | `str` | Inherited |
| `source_format` | `Literal[...]` | Inherited |
| `title` | `str` | Inherited |
| `section_heading` | `str \| None` | Inherited |
| `page_number` | `int \| None` | Inherited |
| `text` | `str` | Chunk text |
| `chunk_index` | `int` | 0-based, continuous across all docs in one `chunk()` call |
| `strategy` | `ChunkingStrategy` | `"fixed_size"`, `"recursive_header"`, or `"semantic"` |
| `processed_at` | `str` | ISO-8601 UTC timestamp |

**Strategies:**

| Strategy | Splitter | Separators | Overlap |
|----------|----------|------------|---------|
| `fixed_size` | `RecursiveCharacterTextSplitter` | Default (`\n\n`, `\n`, ` `, `""`) | Yes |
| `recursive_header` | `RecursiveCharacterTextSplitter` | `\n\n`, `\n`, `. `, `! `, `? `, ` `, `""` | Yes |
| `semantic` | Custom (cosine distance on embeddings) | Sentence boundaries | No |

The semantic strategy calls `openai.embeddings.create` for each multi-sentence document. The OpenAI client is lazily instantiated — single-sentence documents skip the API entirely.

**Public API:**

```python
from src.ingestion import DocumentLoader, Chunker, Chunk, ChunkingStrategy, chunk_id

loader = DocumentLoader(settings)
docs = loader.load(Path("data/raw/guide.pdf"))

chunker = Chunker(settings)          # strategy read from settings.chunk_strategy
chunks: list[Chunk] = chunker.chunk(docs)
```

---

## 2026-06-28 — Phase 1: Document Loader (Complete)

### Module Layout

```
src/
  config.py           # Central settings via pydantic-settings (singleton)
  ingestion/
    models.py         # ProcessedDocument, Chunk, ChunkingStrategy, chunk_id
    chunker.py        # Chunker — three switchable chunking strategies
    loader.py         # DocumentLoader — dispatches by file extension
    storage.py        # save_processed / load_processed / list_raw_files
  retrieval/          # EmbedderProtocol/make_embedder, VectorStoreProtocol/make_vector_store,
                      # BM25Store, Indexer (complete)
                      # providers/        # embedder_openai.py, embedder_sentence_transformers.py,
                      #                   # embedder_voyage.py, embedder_gemini.py, embedder_cohere.py
                      #                   # (all complete; qdrant vector store still planned)
                      # DenseRetriever, SparseRetriever, VectorStoreHit (complete)
                      # reciprocal_rank_fusion, HybridRetriever (complete)
                      # RerankerProtocol/make_reranker, SentenceTransformersReranker (complete)
  generation/         # Grounded prompt, citation parser/verifier, confidence scorer, fallback
                      # response (complete; no generation orchestrator wiring them together yet)
  tracing/            # Trace/Span models (complete); context manager, decorator, JSON+SQLite
                      # writers [planned]
  analysis/           # Backward trace walker, failure categorizer, evidence chain builder [planned]
  evaluation/         # Golden dataset runner, metric calculators, regression tracker [planned]
  api/                # FastAPI app, route handlers [planned]
  frontend/           # Streamlit or React query dashboard and trace explorer [planned]
scripts/
  seed_corpus.py      # Index sample docs for local testing
  run_eval.py         # Execute full eval suite and print metrics
tests/
  fixtures/           # Sample files (sample.md, sample.txt, sample.html) + PDF generator
  unit/ingestion/     # Unit tests for models, loader, storage
  unit/retrieval/     # Unit tests for Embedder, VectorStore, BM25Store, Indexer
  integration/        # End-to-end pipeline tests
data/
  raw/                # Source documents (original, untouched)
  processed/          # Normalised plaintext + metadata (one JSON per section/page)
  chroma/             # ChromaDB file-based persistence
  bm25_index.pkl      # BM25 index (pickled corpus + chunk_ids, rebuilt on load)
  traces/             # JSON trace files (one per request) [planned]
  eval/               # Golden Q&A dataset and flagged failure cases [planned]
```

### Ingestion Module

The ingestion module (`src/ingestion/`) is the entry point for all content. It accepts `.md`, `.txt`, `.html`, and `.pdf` files, normalises them to clean plaintext, and attaches structured metadata.

**`ProcessedDocument`** is the single output type for all formats:

| Field | Type | Description |
|-------|------|-------------|
| `doc_id` | `str` | SHA-256 of raw file bytes — stable across re-ingestion |
| `source_path` | `str` | Path to the original file |
| `source_format` | `Literal[...]` | `"markdown"`, `"text"`, `"html"`, or `"pdf"` |
| `title` | `str` | First heading found, or filename stem |
| `section_heading` | `str \| None` | Nearest preceding heading; `None` for plain text and PDF |
| `page_number` | `int \| None` | 1-indexed page number for PDF; `None` otherwise |
| `text` | `str` | Clean plaintext — no markup |
| `processed_at` | `str` | ISO-8601 UTC timestamp |

**`DocumentLoader.load(path)`** dispatches by file extension:

| Format | Splits on | `section_heading` | `page_number` |
|--------|-----------|-------------------|---------------|
| `.md` | `#` / `##` / `###` headings | Nearest preceding heading | `None` |
| `.txt` | Whole file | `None` | `None` |
| `.html` / `.htm` | `<h1>`–`<h6>` tags | Nearest preceding heading | `None` |
| `.pdf` | Pages | `None` | 1-indexed |

**Storage** mirrors the source path under `data/processed/`:

```
data/raw/guide.pdf          → data/processed/guide.pdf/page_001.json
data/raw/setup.md           → data/processed/setup.md/section_000.json
```

Re-ingesting a file overwrites its output directory. Because `doc_id` derives from raw bytes, the ID is identical every time the same file is loaded — safe for downstream deduplication checks.

**Public API:**

```python
from src.ingestion import DocumentLoader, Chunker, save_processed, load_processed, list_raw_files

loader = DocumentLoader(settings)
docs = loader.load(Path("data/raw/guide.pdf"))
save_processed(docs, source_raw_path, settings.processed_data_dir)

# Re-index without re-upload:
docs = load_processed(source_raw_path, settings.processed_data_dir)
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
