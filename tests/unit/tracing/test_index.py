from __future__ import annotations

from datetime import UTC, datetime

from src.tracing.index import (
    get_trace_record,
    index_trace,
    init_trace_index,
    list_trace_records,
)
from src.tracing.models import Trace


def _trace(**overrides: object) -> Trace:
    base: dict[str, object] = {"status": "success"}
    base.update(overrides)
    return Trace(**base)


class TestInitTraceIndex:
    def test_creates_db_file(self, tmp_path):
        db_path = tmp_path / "traces.db"
        init_trace_index(db_path)
        assert db_path.exists()

    def test_idempotent_when_called_twice(self, tmp_path):
        db_path = tmp_path / "traces.db"
        init_trace_index(db_path)
        init_trace_index(db_path)  # must not raise

    def test_creates_parent_dir_when_missing(self, tmp_path):
        db_path = tmp_path / "nested" / "traces.db"
        init_trace_index(db_path)
        assert db_path.exists()


class TestIndexTrace:
    def test_row_present_after_indexing(self, tmp_path):
        db_path = tmp_path / "traces.db"
        trace = _trace(final_score=0.75)
        trace_path = tmp_path / f"{trace.trace_id}.json"
        index_trace(trace, db_path, trace_path)

        record = get_trace_record(trace.trace_id, db_path)
        assert record is not None
        assert record.trace_id == trace.trace_id
        assert record.status == "success"
        assert record.final_score == 0.75
        assert record.trace_path == str(trace_path)

    def test_reindexing_same_trace_id_updates_row(self, tmp_path):
        db_path = tmp_path / "traces.db"
        trace = _trace(status="degraded", final_score=0.4)
        trace_path = tmp_path / f"{trace.trace_id}.json"
        index_trace(trace, db_path, trace_path)

        updated = trace.model_copy(update={"status": "success", "final_score": 0.9})
        index_trace(updated, db_path, trace_path)

        records = list_trace_records(db_path)
        matching = [r for r in records if r.trace_id == trace.trace_id]
        assert len(matching) == 1
        assert matching[0].status == "success"
        assert matching[0].final_score == 0.9

    def test_final_score_none_stored_as_null(self, tmp_path):
        db_path = tmp_path / "traces.db"
        trace = _trace()
        trace_path = tmp_path / f"{trace.trace_id}.json"
        index_trace(trace, db_path, trace_path)

        record = get_trace_record(trace.trace_id, db_path)
        assert record is not None
        assert record.final_score is None


class TestGetTraceRecord:
    def test_unknown_trace_id_returns_none(self, tmp_path):
        db_path = tmp_path / "traces.db"
        init_trace_index(db_path)
        assert get_trace_record("does-not-exist", db_path) is None


class TestListTraceRecords:
    def test_orders_by_timestamp_desc(self, tmp_path):
        db_path = tmp_path / "traces.db"
        older = _trace(timestamp=datetime(2026, 1, 1, tzinfo=UTC))
        newer = _trace(timestamp=datetime(2026, 6, 1, tzinfo=UTC))
        index_trace(older, db_path, tmp_path / f"{older.trace_id}.json")
        index_trace(newer, db_path, tmp_path / f"{newer.trace_id}.json")

        records = list_trace_records(db_path)
        assert [r.trace_id for r in records] == [newer.trace_id, older.trace_id]

    def test_limit_truncates_results(self, tmp_path):
        db_path = tmp_path / "traces.db"
        for _ in range(3):
            t = _trace()
            index_trace(t, db_path, tmp_path / f"{t.trace_id}.json")

        records = list_trace_records(db_path, limit=2)
        assert len(records) == 2

    def test_status_filter_returns_only_matching(self, tmp_path):
        db_path = tmp_path / "traces.db"
        success = _trace(status="success")
        failure = _trace(status="failure")
        index_trace(success, db_path, tmp_path / f"{success.trace_id}.json")
        index_trace(failure, db_path, tmp_path / f"{failure.trace_id}.json")

        records = list_trace_records(db_path, status="failure")
        assert [r.trace_id for r in records] == [failure.trace_id]

    def test_no_filter_returns_all(self, tmp_path):
        db_path = tmp_path / "traces.db"
        success = _trace(status="success")
        failure = _trace(status="failure")
        index_trace(success, db_path, tmp_path / f"{success.trace_id}.json")
        index_trace(failure, db_path, tmp_path / f"{failure.trace_id}.json")

        records = list_trace_records(db_path)
        assert {r.trace_id for r in records} == {success.trace_id, failure.trace_id}
