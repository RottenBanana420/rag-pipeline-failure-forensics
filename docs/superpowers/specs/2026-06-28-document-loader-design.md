# Document Loader Design

**Date:** 2026-06-28
**Phase:** 1 — Ingestion
**Status:** Approved

## Problem

The RAG pipeline needs a reliable entry point for documents. Files arrive in multiple formats (Markdown, plain text, HTML, PDF) and must be normalised to clean plaintext with consistent metadata before chunking, embedding, and indexing can occur. Processed output must be stored alongside raw files so the system can re-index without requiring re-upload.

## Scope

Implements `src/ingestion/` — specifically the `models`, `loader`, and `storage` modules. Does not include chunking, embedding, or deduplication (Phase 2).

## Data Model

```python
class ProcessedDocument(BaseModel):
    doc_id: str           # SHA-256 of raw file bytes — stable across re-ingestion
    source_path: str      # Path relative to data/raw/, e.g. "guides/api.pdf"
    source_format: str    # Literal["markdown", "text", "html", "pdf"]
    title: str            # First heading found, or filename stem if none
    section_heading: str | None  # Nearest preceding heading; None for txt and pdf
    page_number: int | None      # 1-indexed page number for PDF; None otherwise
    text: str             # Clean plaintext — no markup, no boilerplate
    processed_at: str     # ISO-8601 UTC timestamp
```

`doc_id` is computed from raw bytes before any processing. All documents produced from the same file share the same `doc_id`, which makes deduplication checks in later phases trivial.

## Module Layout

```
src/ingestion/
  __init__.py     # Public exports
  models.py       # ProcessedDocument Pydantic model
  loader.py       # DocumentLoader class
  storage.py      # Persistence functions
```

## Loader Behaviour Per Format

| Format | Library | Splits on | section_heading | page_number |
|--------|---------|-----------|-----------------|-------------|
| `.md` | `markdown-it-py` | Headings (`#`/`##`/`###`) | Nearest preceding heading | None |
| `.txt` | stdlib | Whole file | None | None |
| `.html` | `beautifulsoup4` | `<h1>`–`<h6>` tags | Nearest preceding heading | None |
| `.pdf` | `pypdf` | Pages | None | 1-indexed |

For Markdown and HTML: one `ProcessedDocument` per section. For plain text: one document per file. For PDF: one document per page. An introductory section before the first heading in Markdown/HTML is included as its own document with `section_heading=None`.

## Error Handling

- **Unsupported extension:** `ValueError` raised immediately.
- **Empty file:** returns `[]`, no error.
- **Corrupt/unreadable file:** underlying library exception propagates; callers decide.

## Storage Layout

```
data/raw/guides/api.pdf           ← original, untouched
data/processed/guides/api.pdf/
    page_001.json                 ← one ProcessedDocument per page (PDF)
    page_002.json

data/raw/guides/setup.md          ← original
data/processed/guides/setup.md/
    section_000.json              ← one ProcessedDocument per section
    section_001.json
```

Re-ingesting a file overwrites its `data/processed/` directory entirely.

## Public API

```python
from src.ingestion import DocumentLoader, save_processed, load_processed, list_raw_files

loader = DocumentLoader(settings)
docs = loader.load(Path("data/raw/guides/api.pdf"))
save_processed(docs, source_raw_path, settings.processed_data_dir)

# Later — re-index without re-upload:
docs = load_processed(source_raw_path, settings.processed_data_dir)
```

## Configuration

Two fields added to `src/config.py`:
- `raw_data_dir: Path` — default `./data/raw`
- `processed_data_dir: Path` — default `./data/processed`

## Testing

- Unit tests in `tests/unit/ingestion/` per module.
- Small fixture files in `tests/fixtures/` (`.md`, `.txt`, `.html`; PDF generated in conftest).
- Integration roundtrip test: load → save → load_processed → assert field equality.
