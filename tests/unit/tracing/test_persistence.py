from __future__ import annotations

from src.config import Settings
from src.tracing.index import get_trace_record
from src.tracing.models import Span, Trace
from src.tracing.persistence import persist_trace
from src.tracing.storage import load_trace


def _settings(tmp_path) -> Settings:
    return Settings(
        trace_output_dir=tmp_path / "traces",
        sqlite_db_path=tmp_path / "traces.db",
    )


class TestPersistTrace:
    def test_returns_path_to_written_json_file(self, tmp_path):
        settings = _settings(tmp_path)
        trace = Trace(status="success")

        path = persist_trace(trace, settings)

        assert path == settings.trace_output_dir / f"{trace.trace_id}.json"
        assert path.exists()

    def test_written_json_round_trips(self, tmp_path):
        settings = _settings(tmp_path)
        span = Span(step="generation", input="{}", output="answer", latency_ms=2.0)
        trace = Trace(status="success", spans=[span], final_score=0.7)

        persist_trace(trace, settings)

        restored = load_trace(trace.trace_id, settings.trace_output_dir)
        assert restored == trace

    def test_indexes_trace_metadata_in_sqlite(self, tmp_path):
        settings = _settings(tmp_path)
        trace = Trace(status="degraded", final_score=0.3)

        path = persist_trace(trace, settings)

        record = get_trace_record(trace.trace_id, settings.sqlite_db_path)
        assert record is not None
        assert record.status == "degraded"
        assert record.final_score == 0.3
        assert record.trace_path == str(path)
