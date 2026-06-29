from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

from src.ingestion.models import ProcessedDocument

_SUPPORTED_EXTENSIONS = {".md", ".txt", ".html", ".htm", ".pdf"}


def save_processed(
    docs: list[ProcessedDocument],
    source_raw_path: Path,
    processed_dir: Path,
) -> None:
    """Write one JSON file per ProcessedDocument under processed_dir.

    The output directory mirrors source_raw_path:
        data/processed/guides/api.pdf/page_001.json

    Re-ingesting the same file overwrites the directory entirely.
    """
    out_dir = _output_dir(source_raw_path, processed_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    is_pdf = source_raw_path.suffix.lower() == ".pdf"

    for i, doc in enumerate(docs):
        if is_pdf and doc.page_number is not None:
            filename = f"page_{doc.page_number:03d}.json"
        else:
            filename = f"section_{i:03d}.json"
        (out_dir / filename).write_text(
            doc.model_dump_json(indent=2), encoding="utf-8"
        )


def load_processed(
    source_raw_path: Path,
    processed_dir: Path,
) -> list[ProcessedDocument]:
    """Read all ProcessedDocument JSON files for a given source file."""
    out_dir = _output_dir(source_raw_path, processed_dir)
    if not out_dir.exists():
        return []

    docs: list[ProcessedDocument] = []
    for json_file in sorted(out_dir.glob("*.json")):
        docs.append(ProcessedDocument.model_validate_json(json_file.read_text("utf-8")))
    return docs


def list_raw_files(raw_dir: Path) -> Iterator[Path]:
    """Yield all files in raw_dir with a supported extension."""
    for path in sorted(raw_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in _SUPPORTED_EXTENSIONS:
            yield path


# ------------------------------------------------------------------
# Internal
# ------------------------------------------------------------------

def _output_dir(source_raw_path: Path, processed_dir: Path) -> Path:
    """Return the processed output directory for a given source file.

    Mirrors the source path relative to its own parent structure:
        data/raw/guides/api.pdf  →  data/processed/guides/api.pdf/
    """
    # Use only the filename + one level of parent to keep paths portable
    relative = Path(source_raw_path.name)
    return processed_dir / relative
