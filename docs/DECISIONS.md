# Architecture Decision Records

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
