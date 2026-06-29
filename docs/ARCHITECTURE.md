# Architecture Overview

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
  retrieval/          # Dense (ChromaDB), sparse (BM25), RRF fusion, reranker [planned]
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
  unit/retrieval/     # [planned]
  integration/        # End-to-end pipeline tests
data/
  raw/                # Source documents (original, untouched)
  processed/          # Normalised plaintext + metadata (one JSON per section/page)
  traces/             # JSON trace files (one per request) [planned]
  eval/               # Golden Q&A dataset and flagged failure cases [planned]
  chroma/             # ChromaDB file-based persistence [planned]
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
