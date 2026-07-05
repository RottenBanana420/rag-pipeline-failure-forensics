from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from src.tracing.models import Trace, TraceStatus

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id    TEXT PRIMARY KEY,
    timestamp   TEXT NOT NULL,
    status      TEXT NOT NULL,
    final_score REAL,
    trace_path  TEXT NOT NULL
)
"""

_CREATE_TIMESTAMP_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_traces_timestamp ON traces (timestamp DESC)"
)

_INSERT_SQL = """
INSERT OR REPLACE INTO traces (trace_id, timestamp, status, final_score, trace_path)
VALUES (?, ?, ?, ?, ?)
"""

_SELECT_ONE_SQL = """
SELECT trace_id, timestamp, status, final_score, trace_path
FROM traces WHERE trace_id = ?
"""


class TraceRecord(BaseModel):
    trace_id: str
    timestamp: datetime
    status: TraceStatus
    final_score: float | None
    trace_path: str


@contextmanager
def _connection(db_path: Path) -> Iterator[sqlite3.Connection]:
    """One unit of work: commits on success, rolls back on exception, and
    always closes the connection afterward (sqlite3's `with conn:` handles
    commit/rollback but does not close the connection on its own).
    """
    conn = sqlite3.connect(db_path)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_trace_index(db_path: Path) -> None:
    """Create the traces table/index if missing. Idempotent.

    Creates db_path's parent directory lazily if missing.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _connection(db_path) as conn:
        conn.execute(_CREATE_TABLE_SQL)
        conn.execute(_CREATE_TIMESTAMP_INDEX_SQL)


def index_trace(trace: Trace, db_path: Path, trace_path: Path) -> None:
    """Insert or update trace's metadata row, keyed on trace_id.

    INSERT OR REPLACE makes re-indexing the same trace_id idempotent.
    """
    init_trace_index(db_path)
    with _connection(db_path) as conn:
        conn.execute(
            _INSERT_SQL,
            (
                trace.trace_id,
                trace.timestamp.isoformat(),
                trace.status,
                trace.final_score,
                str(trace_path),
            ),
        )


def get_trace_record(trace_id: str, db_path: Path) -> TraceRecord | None:
    with _connection(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(_SELECT_ONE_SQL, (trace_id,)).fetchone()
    return TraceRecord(**dict(row)) if row is not None else None


def list_trace_records(
    db_path: Path,
    status: TraceStatus | None = None,
    limit: int = 100,
) -> list[TraceRecord]:
    where_clause = "WHERE status = ?" if status is not None else ""
    sql = (
        "SELECT trace_id, timestamp, status, final_score, trace_path "
        f"FROM traces {where_clause} ORDER BY timestamp DESC LIMIT ?"
    )
    params: tuple[object, ...] = (status, limit) if status is not None else (limit,)
    with _connection(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
    return [TraceRecord(**dict(row)) for row in rows]
