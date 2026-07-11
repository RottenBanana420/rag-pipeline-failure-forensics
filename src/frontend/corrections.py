"""Persistence for human-entered per-span "expected output" corrections.

One JSON file per trace (`{corrections_dir}/{trace_id}.json`), mapping
span_id -> expected_output. Mirrors `src.tracing.storage.save_trace`'s
one-file-per-id convention, but keyed by trace_id with span_id as a nested
key since corrections are entered one span at a time within a trace.
"""

from __future__ import annotations

import json
from pathlib import Path


def _corrections_path(trace_id: str, corrections_dir: Path) -> Path:
    return corrections_dir / f"{trace_id}.json"


def save_correction(
    trace_id: str, span_id: str, expected_output: str, corrections_dir: Path
) -> None:
    """Persist *expected_output* as the human correction for *span_id*.

    Creates corrections_dir lazily if missing. Preserves any other spans'
    corrections already saved for this trace.
    """
    corrections_dir.mkdir(parents=True, exist_ok=True)
    path = _corrections_path(trace_id, corrections_dir)
    corrections = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    corrections[span_id] = expected_output
    path.write_text(json.dumps(corrections, indent=2), encoding="utf-8")


def load_correction(trace_id: str, span_id: str, corrections_dir: Path) -> str | None:
    """Return the saved human correction for *span_id*, or None if absent."""
    path = _corrections_path(trace_id, corrections_dir)
    if not path.exists():
        return None
    corrections: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))
    return corrections.get(span_id)
