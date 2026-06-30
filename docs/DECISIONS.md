# Architecture Decision Records

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

## 2026-06-30 — Hybrid Retrieval Query Path

**`DenseRetriever` wraps `Embedder` + `VectorStore.query`** — The retriever is a thin stateless facade: embed query → query collection. Keeping embedding and querying separate (rather than adding a `query_text` method to `VectorStore`) means tests can swap `Embedder` for a fixture without touching `VectorStore`, and the same `Embedder` instance is shared across the indexing and query paths with no duplication.

**`SparseRetriever` fetches metadata from `VectorStore`, not `BM25Store`** — BM25 stores only `chunk_ids` and `corpus` (raw tokens) for index construction. Duplicating full chunk metadata (text, title, source_path, etc.) into the BM25 pickle would double storage and create a sync hazard. Instead, `SparseRetriever` calls `VectorStore.get_by_ids(ids)` to hydrate results — one ChromaDB round-trip per query regardless of `k`.

**BM25 scores normalized by max score before returning** — Raw BM25 scores are unbounded and dataset-dependent. Dividing by `max_score` maps results to (0, 1], making the `similarity` field semantically comparable to cosine similarity. This is required for RRF fusion (next step) and for displaying consistent confidence scores across retrieval modes.

**Zero-score guard in `SparseRetriever`** — After sorting, chunks with `score == 0.0` are dropped before normalization. A zero-score chunk means BM25 found no query term overlap — including it would produce `0 / max_score = 0` similarity entries that pollute fusion results. The guard also prevents division-by-zero when all scores are zero (empty index or no term overlap).

**`VectorStoreHit` as the shared result model** — Both `DenseRetriever` and `SparseRetriever` return `list[VectorStoreHit]`. Using a single frozen dataclass means the RRF fusion layer and reranker don't need to know which retriever produced a hit. `dataclasses.replace` in `SparseRetriever` swaps in the normalized BM25 score without mutating the ChromaDB-returned hit.
