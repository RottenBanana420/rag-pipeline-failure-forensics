"""Unit tests for trace-level "bad output" flag persistence."""

from __future__ import annotations

import json

from src.analysis.evidence_chain import EvidenceChain, EvidenceEntry
from src.analysis.failure_categorizer import FailureCategoryVerdict
from src.analysis.root_cause import RootCauseDiagnosis
from src.frontend.diagnosis_service import DiagnosisResult
from src.frontend.flags import (
    DiagnosisSummary,
    FlagRecord,
    HumanReview,
    diagnosis_summary_from_result,
    load_flag,
    save_flag,
)
from src.tracing.models import Span


def make_span(step="retrieval", input="in", output="out", **overrides) -> Span:
    base: dict[str, object] = {
        "step": step,
        "input": input,
        "output": output,
        "latency_ms": 1.0,
    }
    base.update(overrides)
    return Span(**base)


def make_diagnosis_summary(**overrides: object) -> DiagnosisSummary:
    base: dict[str, object] = {
        "root_cause_span_id": "span-1",
        "step": "retrieval",
        "score": 1,
        "rationale": "No relevant chunks returned.",
        "category": "retrieval_failure",
        "category_rationale": "Query terms never matched the corpus.",
        "narrative": "Retrieval failed to surface the answer.",
    }
    base.update(overrides)
    return DiagnosisSummary(**base)


def make_human_review(**overrides: object) -> HumanReview:
    base: dict[str, object] = {
        "confirmed": True,
        "span_id": "span-1",
        "category": "retrieval_failure",
        "note": "",
    }
    base.update(overrides)
    return HumanReview(**base)


class TestSaveAndLoadFlag:
    def test_round_trip_confirmed(self, tmp_path):
        record = FlagRecord(
            flagged_at="2026-07-12T10:00:00+00:00",
            diagnosis=make_diagnosis_summary(),
            human_review=make_human_review(confirmed=True),
        )

        save_flag("trace-1", record, tmp_path)

        assert load_flag("trace-1", tmp_path) == record

    def test_round_trip_overridden(self, tmp_path):
        record = FlagRecord(
            flagged_at="2026-07-12T10:00:00+00:00",
            diagnosis=make_diagnosis_summary(),
            human_review=make_human_review(
                confirmed=False,
                span_id="span-2",
                category="generation_incomplete",
                note="Actually the answer was cut off mid-sentence.",
            ),
        )

        save_flag("trace-1", record, tmp_path)

        assert load_flag("trace-1", tmp_path) == record

    def test_round_trip_no_diagnosis(self, tmp_path):
        record = FlagRecord(
            flagged_at="2026-07-12T10:00:00+00:00",
            diagnosis=None,
            human_review=make_human_review(
                confirmed=False, span_id="span-1", category="other", note="Bad answer."
            ),
        )

        save_flag("trace-1", record, tmp_path)

        assert load_flag("trace-1", tmp_path) == record

    def test_missing_file_returns_none(self, tmp_path):
        assert load_flag("no-such-trace", tmp_path) is None

    def test_overwrite_updates_existing_flag(self, tmp_path):
        first = FlagRecord(
            flagged_at="2026-07-12T10:00:00+00:00",
            diagnosis=make_diagnosis_summary(),
            human_review=make_human_review(confirmed=True),
        )
        second = FlagRecord(
            flagged_at="2026-07-12T11:00:00+00:00",
            diagnosis=make_diagnosis_summary(),
            human_review=make_human_review(
                confirmed=False, category="context_loss", note="Redid the review."
            ),
        )

        save_flag("trace-1", first, tmp_path)
        save_flag("trace-1", second, tmp_path)

        assert load_flag("trace-1", tmp_path) == second

    def test_different_traces_are_independent(self, tmp_path):
        record_one = FlagRecord(
            flagged_at="2026-07-12T10:00:00+00:00",
            diagnosis=make_diagnosis_summary(),
            human_review=make_human_review(confirmed=True),
        )
        record_two = FlagRecord(
            flagged_at="2026-07-12T10:05:00+00:00",
            diagnosis=None,
            human_review=make_human_review(confirmed=False, note="Different trace."),
        )

        save_flag("trace-1", record_one, tmp_path)
        save_flag("trace-2", record_two, tmp_path)

        assert load_flag("trace-1", tmp_path) == record_one
        assert load_flag("trace-2", tmp_path) == record_two


class TestDiagnosisSummaryFromResult:
    def test_returns_none_when_no_root_cause(self):
        result = DiagnosisResult(diagnosis=None, category=None, evidence_chain=None)

        assert diagnosis_summary_from_result(result) is None

    def test_flattens_full_result(self):
        root_cause_span = make_span(
            step="retrieval", input="query", output="no good chunks", span_id="span-1"
        )
        diagnosis = RootCauseDiagnosis(
            root_cause_span=root_cause_span,
            score=1,
            rationale="Unreasonable retrieval.",
            evaluated_spans=[],
        )
        category = FailureCategoryVerdict(
            category="retrieval_failure", rationale="No relevant chunks."
        )
        evidence_chain = EvidenceChain(
            narrative="Retrieval failed first.",
            category="retrieval_failure",
            category_rationale="No relevant chunks.",
            evidence=[
                EvidenceEntry(
                    step="retrieval",
                    input="query",
                    output="no good chunks",
                    score=1,
                    rationale="Unreasonable retrieval.",
                )
            ],
        )
        result = DiagnosisResult(
            diagnosis=diagnosis, category=category, evidence_chain=evidence_chain
        )

        summary = diagnosis_summary_from_result(result)

        assert summary == DiagnosisSummary(
            root_cause_span_id="span-1",
            step="retrieval",
            score=1,
            rationale="Unreasonable retrieval.",
            category="retrieval_failure",
            category_rationale="No relevant chunks.",
            narrative="Retrieval failed first.",
        )


class TestSerializationShape:
    def test_flag_record_survives_json_round_trip(self, tmp_path):
        record = FlagRecord(
            flagged_at="2026-07-12T10:00:00+00:00",
            diagnosis=make_diagnosis_summary(),
            human_review=make_human_review(confirmed=True),
        )

        save_flag("trace-1", record, tmp_path)
        on_disk = json.loads((tmp_path / "trace-1.json").read_text(encoding="utf-8"))

        assert on_disk["flagged_at"] == "2026-07-12T10:00:00+00:00"
        assert on_disk["diagnosis"]["root_cause_span_id"] == "span-1"
        assert on_disk["human_review"]["confirmed"] is True
        assert load_flag("trace-1", tmp_path) == record

    def test_diagnosis_none_serializes_as_json_null(self, tmp_path):
        record = FlagRecord(
            flagged_at="2026-07-12T10:00:00+00:00",
            diagnosis=None,
            human_review=make_human_review(
                confirmed=False, note="No root cause found."
            ),
        )

        save_flag("trace-1", record, tmp_path)
        on_disk = json.loads((tmp_path / "trace-1.json").read_text(encoding="utf-8"))

        assert on_disk["diagnosis"] is None
