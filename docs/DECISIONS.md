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
