from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.tracing.models import Span


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

    def test_round_trip_json(self):
        span = Span(**_valid_span_kwargs(confidence_score=5, token_count=100))
        restored = Span.model_validate_json(span.model_dump_json())
        assert restored == span
