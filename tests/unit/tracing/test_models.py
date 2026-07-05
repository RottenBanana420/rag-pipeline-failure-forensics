from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from src.tracing.models import Span, Trace


def _valid_span_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "step": "retrieval",
        "input": '{"question": "What is RRF?"}',
        "output": '{"hits": []}',
        "latency_ms": 12.5,
    }
    base.update(overrides)
    return base


class TestSpanValidation:
    def test_valid_minimal_span(self):
        span = Span(**_valid_span_kwargs())
        assert span.step == "retrieval"
        assert span.llm_prompt is None
        assert span.token_count is None
        assert span.confidence_score is None
        assert isinstance(span.span_id, str) and span.span_id

    def test_span_id_auto_generated_and_unique(self):
        span_a = Span(**_valid_span_kwargs())
        span_b = Span(**_valid_span_kwargs())
        assert span_a.span_id != span_b.span_id

    def test_all_pipeline_steps_accepted(self):
        for step in (
            "ingestion",
            "retrieval",
            "ranking",
            "generation",
            "verification",
        ):
            span = Span(**_valid_span_kwargs(step=step))
            assert span.step == step

    def test_invalid_step_rejected(self):
        with pytest.raises(ValidationError):
            Span(**_valid_span_kwargs(step="not_a_step"))

    def test_optional_fields_populated(self):
        span = Span(
            **_valid_span_kwargs(
                step="generation",
                llm_prompt="Answer using only the provided context.",
                token_count=250,
                confidence_score=4,
            )
        )
        assert span.llm_prompt == "Answer using only the provided context."
        assert span.token_count == 250
        assert span.confidence_score == 4

    def test_confidence_score_boundaries_accepted(self):
        assert Span(**_valid_span_kwargs(confidence_score=1)).confidence_score == 1
        assert Span(**_valid_span_kwargs(confidence_score=5)).confidence_score == 5

    def test_confidence_score_below_range_rejected(self):
        with pytest.raises(ValidationError):
            Span(**_valid_span_kwargs(confidence_score=0))

    def test_confidence_score_above_range_rejected(self):
        with pytest.raises(ValidationError):
            Span(**_valid_span_kwargs(confidence_score=6))

    def test_negative_token_count_rejected(self):
        with pytest.raises(ValidationError):
            Span(**_valid_span_kwargs(token_count=-1))

    def test_negative_latency_rejected(self):
        with pytest.raises(ValidationError):
            Span(**_valid_span_kwargs(latency_ms=-0.1))

    def test_empty_input_and_output_accepted(self):
        span = Span(**_valid_span_kwargs(input="", output=""))
        assert span.input == ""
        assert span.output == ""

    def test_error_defaults_to_none(self):
        span = Span(**_valid_span_kwargs())
        assert span.error is None

    def test_error_field_populated(self):
        span = Span(**_valid_span_kwargs(error="RuntimeError: boom"))
        assert span.error == "RuntimeError: boom"

    def test_round_trip_json(self):
        span = Span(**_valid_span_kwargs(confidence_score=5, token_count=100))
        restored = Span.model_validate_json(span.model_dump_json())
        assert restored == span


def _valid_trace_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {"status": "success"}
    base.update(overrides)
    return base


class TestTraceValidation:
    def test_valid_minimal_trace(self):
        trace = Trace(**_valid_trace_kwargs())
        assert trace.status == "success"
        assert trace.spans == []
        assert trace.final_output is None
        assert isinstance(trace.trace_id, str) and trace.trace_id

    def test_trace_id_auto_generated_and_unique(self):
        trace_a = Trace(**_valid_trace_kwargs())
        trace_b = Trace(**_valid_trace_kwargs())
        assert trace_a.trace_id != trace_b.trace_id

    def test_all_statuses_accepted(self):
        for status in ("success", "failure", "degraded"):
            trace = Trace(**_valid_trace_kwargs(status=status))
            assert trace.status == status

    def test_invalid_status_rejected(self):
        with pytest.raises(ValidationError):
            Trace(**_valid_trace_kwargs(status="not_a_status"))

    def test_status_required(self):
        with pytest.raises(ValidationError):
            Trace(spans=[], final_output=None)

    def test_trace_holds_spans(self):
        span = Span(
            step="ingestion",
            input='{"path": "doc.md"}',
            output='{"doc_id": "abc"}',
            latency_ms=3.0,
        )
        trace = Trace(
            **_valid_trace_kwargs(spans=[span], final_output="The answer is 42.")
        )
        assert trace.spans == [span]
        assert trace.final_output == "The answer is 42."

    def test_round_trip_json_with_spans(self):
        span = Span(
            step="generation",
            input='{"prompt": "..."}',
            output="The answer is 42.",
            llm_prompt="Answer using only the provided context.",
            token_count=300,
            latency_ms=850.2,
            confidence_score=4,
        )
        trace = Trace(spans=[span], final_output="The answer is 42.", status="success")
        restored = Trace.model_validate_json(trace.model_dump_json())
        assert restored == trace

    def test_round_trip_model_dump(self):
        span = Span(step="ranking", input="[]", output="[]", latency_ms=5.0)
        trace = Trace(spans=[span], status="degraded")
        restored = Trace.model_validate(trace.model_dump())
        assert restored == trace

    def test_timestamp_auto_generated_and_utc_aware(self):
        trace = Trace(**_valid_trace_kwargs())
        assert trace.timestamp.tzinfo is not None
        now = datetime.now(UTC)
        assert now - trace.timestamp < timedelta(seconds=5)

    def test_timestamp_explicit_value_accepted(self):
        ts = datetime(2026, 1, 1, tzinfo=UTC)
        trace = Trace(**_valid_trace_kwargs(timestamp=ts))
        assert trace.timestamp == ts

    def test_final_score_defaults_to_none(self):
        trace = Trace(**_valid_trace_kwargs())
        assert trace.final_score is None

    def test_final_score_accepts_out_of_unit_range(self):
        assert Trace(**_valid_trace_kwargs(final_score=1.4)).final_score == 1.4
        assert Trace(**_valid_trace_kwargs(final_score=-0.2)).final_score == -0.2

    def test_round_trip_json_with_timestamp_and_final_score(self):
        trace = Trace(**_valid_trace_kwargs(final_score=0.82))
        restored = Trace.model_validate_json(trace.model_dump_json())
        assert restored == trace
        assert restored.final_score == 0.82
