from __future__ import annotations

from pathlib import Path

from src.tracing.models import Trace


def save_trace(trace: Trace, output_dir: Path) -> Path:
    """Write `trace` to `{output_dir}/{trace_id}.json`.

    Creates output_dir lazily if missing. Unlike
    `src.ingestion.storage.save_processed`, never deletes or touches any
    other file already in output_dir — each trace is an independent record.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / f"{trace.trace_id}.json"
    trace_path.write_text(trace.model_dump_json(indent=2), encoding="utf-8")
    return trace_path


def load_trace(trace_id: str, output_dir: Path) -> Trace:
    """Read back the Trace previously written by save_trace.

    Raises FileNotFoundError if no such trace file exists in output_dir.
    """
    trace_path = output_dir / f"{trace_id}.json"
    return Trace.model_validate_json(trace_path.read_text(encoding="utf-8"))
