from __future__ import annotations

import time

import pytest

from src.tracing.context import collect_spans
from src.tracing.instrumentation import default_serialize, span


class TestSpanContextManager:
    def test_no_active_sink_is_noop(self):
        with span("retrieval", input="q") as s:
            s.output = "result"
        # No assertion target — just must not raise.

    def test_records_span_into_active_sink(self):
        with collect_spans() as spans, span("retrieval", input="q") as s:
            s.output = "result"

        assert len(spans) == 1
        recorded = spans[0]
        assert recorded.step == "retrieval"
        assert recorded.input == "q"
        assert recorded.output == "result"
        assert recorded.error is None

    def test_measures_latency(self):
        with collect_spans() as spans, span("retrieval", input="q") as s:
            time.sleep(0.01)
            s.output = "result"

        assert spans[0].latency_ms >= 10.0

    def test_captures_llm_prompt_and_token_count(self):
        with collect_spans() as spans, span("verification", input="claim") as s:
            s.llm_prompt = "system + user prompt"
            s.token_count = 42
            s.output = "verdict"

        assert spans[0].llm_prompt == "system + user prompt"
        assert spans[0].token_count == 42

    def test_exception_is_recorded_and_reraised(self):
        with (
            collect_spans() as spans,
            pytest.raises(RuntimeError, match="boom"),
            span("generation", input="q"),
        ):
            raise RuntimeError("boom")

        assert len(spans) == 1
        assert spans[0].error == "RuntimeError: boom"
        assert spans[0].output == ""

    def test_no_sink_still_reraises_exception(self):
        with pytest.raises(RuntimeError, match="boom"), span("generation", input="q"):
            raise RuntimeError("boom")

    def test_multiple_spans_append_in_order(self):
        with collect_spans() as spans:
            with span("retrieval", input="a") as s:
                s.output = "1"
            with span("ranking", input="b") as s:
                s.output = "2"

        assert [s.step for s in spans] == ["retrieval", "ranking"]


class TestDefaultSerialize:
    def test_serializes_plain_dict(self):
        result = default_serialize({"query": "q", "k": 5})
        assert result == '{"query": "q", "k": 5}'

    def test_serializes_dataclass(self):
        from dataclasses import dataclass

        @dataclass
        class Point:
            x: int
            y: int

        result = default_serialize(Point(1, 2))
        assert result == '{"x": 1, "y": 2}'

    def test_serializes_pydantic_model(self):
        from pydantic import BaseModel

        class Verdict(BaseModel):
            supported: bool
            reasoning: str

        result = default_serialize(Verdict(supported=True, reasoning="ok"))
        assert result == '{"supported": true, "reasoning": "ok"}'

    def test_serializes_list_of_dataclasses(self):
        from dataclasses import dataclass

        @dataclass
        class Point:
            x: int

        result = default_serialize([Point(1), Point(2)])
        assert result == '[{"x": 1}, {"x": 2}]'

    def test_falls_back_to_repr_for_unsupported_type(self):
        class Weird:
            def __repr__(self) -> str:
                return "Weird()"

        result = default_serialize(Weird())
        assert result == '"Weird()"'
