from __future__ import annotations

from src.tracing.context import _active_sink, collect_spans
from src.tracing.models import Span


def _span(step: str = "retrieval") -> Span:
    return Span(step=step, input="in", output="out", latency_ms=1.0)


class TestCollectSpans:
    def test_no_active_sink_outside_context(self):
        assert _active_sink() is None

    def test_yields_empty_list_initially(self):
        with collect_spans() as spans:
            assert spans == []

    def test_appended_spans_visible_through_yielded_list(self):
        with collect_spans() as spans:
            spans.append(_span())
            assert len(spans) == 1

    def test_active_sink_is_the_yielded_list(self):
        with collect_spans() as spans:
            sink = _active_sink()
            assert sink is spans

    def test_sink_resets_after_exit(self):
        with collect_spans():
            pass
        assert _active_sink() is None

    def test_sink_resets_after_exception(self):
        try:
            with collect_spans():
                raise ValueError("boom")
        except ValueError:
            pass
        assert _active_sink() is None

    def test_nested_collect_spans_restores_outer_sink(self):
        with collect_spans() as outer:
            outer.append(_span("ingestion"))
            with collect_spans() as inner:
                inner.append(_span("retrieval"))
                assert _active_sink() is inner
            assert _active_sink() is outer
            assert len(outer) == 1
