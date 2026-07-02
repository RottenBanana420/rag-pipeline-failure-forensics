# Architecture Overview

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
| `chroma` (default vector store) | `ChromaVectorStore` | `src/retrieval/vector_store.py` | Nothing — already a base dependency |

`voyage`, `gemini`, `cohere` embedding providers and the `qdrant` vector store are declared in `Settings`'s `Literal` types and `pyproject.toml` optional extras but not yet implemented; selecting them raises `NotImplementedError`/`ValueError` from the factory.

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
                      # providers/        # embedder_openai.py, embedder_sentence_transformers.py
                      #                   # (voyage/gemini/cohere planned)
                      # DenseRetriever, SparseRetriever, VectorStoreHit (complete)
                      # reciprocal_rank_fusion, HybridRetriever (complete)
                      # reranker [planned]
  generation/         # Grounded prompt, citation parser/verifier, confidence scorer [planned]
  tracing/            # Trace/Span models, context manager, decorator, JSON+SQLite writers [planned]
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
