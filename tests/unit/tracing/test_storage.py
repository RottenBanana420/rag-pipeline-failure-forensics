from __future__ import annotations

import pytest

from src.tracing.models import Span, Trace
from src.tracing.storage import load_trace, save_trace


def _trace(**overrides: object) -> Trace:
    base: dict[str, object] = {"status": "success"}
    base.update(overrides)
    return Trace(**base)


class TestSaveTrace:
    def test_writes_json_file_named_by_trace_id(self, tmp_path):
        trace = _trace()
        path = save_trace(trace, tmp_path)
        assert path == tmp_path / f"{trace.trace_id}.json"
        assert path.exists()

    def test_creates_output_dir_when_missing(self, tmp_path):
        output_dir = tmp_path / "nested" / "traces"
        trace = _trace()
        path = save_trace(trace, output_dir)
        assert path.exists()

    def test_written_content_round_trips(self, tmp_path):
        span = Span(step="retrieval", input="{}", output="{}", latency_ms=1.0)
        trace = _trace(spans=[span], final_output="answer", final_score=0.9)
        path = save_trace(trace, tmp_path)
        restored = Trace.model_validate_json(path.read_text(encoding="utf-8"))
        assert restored == trace

    def test_does_not_touch_other_files_in_output_dir(self, tmp_path):
        trace_a = _trace()
        trace_b = _trace()
        path_a = save_trace(trace_a, tmp_path)
        path_b = save_trace(trace_b, tmp_path)
        assert path_a.exists()
        assert path_b.exists()


class TestLoadTrace:
    def test_round_trip_via_save_and_load(self, tmp_path):
        trace = _trace(final_score=0.5)
        save_trace(trace, tmp_path)
        restored = load_trace(trace.trace_id, tmp_path)
        assert restored == trace

    def test_unknown_trace_id_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_trace("does-not-exist", tmp_path)
