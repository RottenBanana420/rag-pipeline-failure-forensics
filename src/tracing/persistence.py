from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from src.tracing.index import index_trace
from src.tracing.models import Trace
from src.tracing.storage import save_trace

if TYPE_CHECKING:
    from src.config import Settings


def persist_trace(trace: Trace, settings: Settings) -> Path:
    """Write trace to JSON and index its metadata in SQLite.

    Standalone entry point: takes an already-assembled Trace rather than
    building one from collect_spans() itself — that's a future
    orchestrator's job. Returns the path of the written JSON trace file.
    """
    trace_path = save_trace(trace, settings.trace_output_dir)
    index_trace(trace, settings.sqlite_db_path, trace_path)
    return trace_path
