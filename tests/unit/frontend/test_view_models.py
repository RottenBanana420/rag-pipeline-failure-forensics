"""Unit tests for the trace-view's pure view-model construction (color-coding
logic and Trace -> TraceGraphViewModel derivation), fully independent of
Streamlit/LLM machinery."""

from __future__ import annotations

import dataclasses

import pytest

from src.analysis.root_cause import RootCauseDiagnosis, SpanQualityResult
from src.frontend.view_models import (
    NODE_STATUS_COLOR,
    build_graph_view_model,
    build_span_diff_view_model,
    cited_chunk_indices,
    node_status,
    root_cause_span_id_from_diagnosis,
)
from src.generation.citation_verifier import CitationVerificationResult
from src.tracing.models import PipelineStep, Span, Trace


def make_span(
    step: PipelineStep = "retrieval",
    input: str = "in",
    output: str = "out",
    **overrides: object,
) -> Span:
    base: dict[str, object] = {
        "step": step,
        "input": input,
        "output": output,
        "latency_ms": 1.0,
    }
    base.update(overrides)
    return Span(**base)


def make_diagnosis(
    span: Span, score: int = 1, rationale: str = "bad"
) -> RootCauseDiagnosis:
    return RootCauseDiagnosis(
        root_cause_span=span,
        score=score,
        rationale=rationale,
        evaluated_spans=[
            SpanQualityResult(span=span, score=score, rationale=rationale)
        ],
    )


class TestNodeStatus:
    def test_root_cause_match_wins_even_with_high_confidence(self):
        span = make_span(confidence_score=5)

        status = node_status(
            span, root_cause_span_id=span.span_id, low_confidence_threshold=2
        )

        assert status == "root_cause"

    def test_confidence_at_or_below_threshold_is_low_confidence(self):
        span = make_span(confidence_score=2)

        status = node_status(span, root_cause_span_id=None, low_confidence_threshold=2)

        assert status == "low_confidence"

    def test_confidence_above_threshold_is_healthy(self):
        span = make_span(confidence_score=3)

        status = node_status(span, root_cause_span_id=None, low_confidence_threshold=2)

        assert status == "healthy"

    def test_none_confidence_is_healthy(self):
        span = make_span(confidence_score=None)

        status = node_status(span, root_cause_span_id=None, low_confidence_threshold=2)

        assert status == "healthy"

    def test_different_span_id_does_not_trigger_root_cause(self):
        span = make_span(confidence_score=5)

        status = node_status(
            span, root_cause_span_id="some-other-span-id", low_confidence_threshold=2
        )

        assert status == "healthy"

    def test_every_status_has_a_color(self):
        for status in ("healthy", "low_confidence", "root_cause"):
            assert status in NODE_STATUS_COLOR
            assert NODE_STATUS_COLOR[status].startswith("#")


class TestBuildGraphViewModel:
    def test_node_count_and_order_matches_spans(self):
        spans = [
            make_span(step="ingestion"),
            make_span(step="retrieval"),
            make_span(step="generation"),
        ]
        trace = Trace(spans=spans, status="success")

        view_model = build_graph_view_model(
            trace, root_cause_span_id=None, low_confidence_threshold=2
        )

        assert [n.span_id for n in view_model.nodes] == [s.span_id for s in spans]
        assert [n.order for n in view_model.nodes] == [0, 1, 2]

    def test_label_includes_position_and_step(self):
        trace = Trace(spans=[make_span(step="ranking")], status="success")

        view_model = build_graph_view_model(
            trace, root_cause_span_id=None, low_confidence_threshold=2
        )

        assert view_model.nodes[0].label == "1. ranking"

    def test_two_spans_sharing_same_step_produce_two_distinct_nodes(self):
        dense = make_span(step="retrieval", input="dense-query")
        sparse = make_span(step="retrieval", input="sparse-query")
        trace = Trace(spans=[dense, sparse], status="success")

        view_model = build_graph_view_model(
            trace, root_cause_span_id=None, low_confidence_threshold=2
        )

        assert len(view_model.nodes) == 2
        assert view_model.nodes[0].span_id != view_model.nodes[1].span_id
        assert view_model.nodes[0].label == "1. retrieval"
        assert view_model.nodes[1].label == "2. retrieval"

    def test_edges_connect_consecutive_spans_only(self):
        spans = [make_span(), make_span(), make_span()]
        trace = Trace(spans=spans, status="success")

        view_model = build_graph_view_model(
            trace, root_cause_span_id=None, low_confidence_threshold=2
        )

        assert view_model.edges == [
            (spans[0].span_id, spans[1].span_id),
            (spans[1].span_id, spans[2].span_id),
        ]
        assert len(view_model.edges) == len(spans) - 1

    def test_empty_spans_produces_empty_nodes_and_edges(self):
        trace = Trace(spans=[], status="success")

        view_model = build_graph_view_model(
            trace, root_cause_span_id=None, low_confidence_threshold=2
        )

        assert view_model.nodes == []
        assert view_model.edges == []

    def test_is_gate_passthrough(self):
        gate_span = make_span(is_gate=True)
        trace = Trace(spans=[gate_span], status="success")

        view_model = build_graph_view_model(
            trace, root_cause_span_id=None, low_confidence_threshold=2
        )

        assert view_model.nodes[0].is_gate is True

    def test_root_cause_span_id_colors_matching_node_red(self):
        spans = [make_span(step="retrieval"), make_span(step="generation")]
        trace = Trace(spans=spans, status="failure")

        view_model = build_graph_view_model(
            trace,
            root_cause_span_id=spans[0].span_id,
            low_confidence_threshold=2,
        )

        assert view_model.nodes[0].status == "root_cause"
        assert view_model.nodes[1].status == "healthy"

    def test_trace_id_passthrough(self):
        trace = Trace(spans=[], status="success", trace_id="my-trace-id")

        view_model = build_graph_view_model(
            trace, root_cause_span_id=None, low_confidence_threshold=2
        )

        assert view_model.trace_id == "my-trace-id"


class TestNodeViewModelAndTraceGraphViewModel:
    def test_node_view_model_is_frozen(self):
        from src.frontend.view_models import NodeViewModel

        node = NodeViewModel(
            span_id="x",
            order=0,
            step="retrieval",
            label="1. retrieval",
            status="healthy",
            is_gate=False,
        )

        with pytest.raises(dataclasses.FrozenInstanceError):
            node.order = 1  # type: ignore[misc]

    def test_trace_graph_view_model_is_frozen(self):
        from src.frontend.view_models import TraceGraphViewModel

        vm = TraceGraphViewModel(trace_id="t", nodes=[], edges=[])

        with pytest.raises(dataclasses.FrozenInstanceError):
            vm.trace_id = "other"  # type: ignore[misc]


class TestRootCauseSpanIdFromDiagnosis:
    def test_none_diagnosis_returns_none(self):
        assert root_cause_span_id_from_diagnosis(None) is None

    def test_real_diagnosis_returns_its_span_id(self):
        span = make_span()
        diagnosis = make_diagnosis(span)

        assert root_cause_span_id_from_diagnosis(diagnosis) == span.span_id


class TestBuildSpanDiffViewModel:
    def test_no_expected_output_yields_no_segments(self):
        span = make_span(output="the sky is blue")

        vm = build_span_diff_view_model(span, None)

        assert vm.expected is None
        assert vm.expected_segments is None
        assert vm.produced_segments is None
        assert vm.received == span.input
        assert vm.produced == span.output

    def test_identical_expected_and_produced_are_all_equal(self):
        span = make_span(output="the sky is blue")

        vm = build_span_diff_view_model(span, "the sky is blue")

        assert vm.expected_segments is not None
        assert vm.produced_segments is not None
        assert all(s.tag == "equal" for s in vm.expected_segments)
        assert all(s.tag == "equal" for s in vm.produced_segments)

    def test_changed_word_tags_each_side(self):
        span = make_span(output="the sky is green")

        vm = build_span_diff_view_model(span, "the sky is blue")

        assert vm.expected_segments is not None
        assert vm.produced_segments is not None
        expected_tagged = {s.tag for s in vm.expected_segments}
        produced_tagged = {s.tag for s in vm.produced_segments}
        assert "expected_only" in expected_tagged
        assert "produced_only" in produced_tagged
        assert any(
            s.tag == "expected_only" and "blue" in s.text for s in vm.expected_segments
        )
        assert any(
            s.tag == "produced_only" and "green" in s.text for s in vm.produced_segments
        )

    def test_added_word_only_marked_on_produced_side(self):
        span = make_span(output="the sky is very blue")

        vm = build_span_diff_view_model(span, "the sky is blue")

        assert vm.expected_segments is not None
        assert vm.produced_segments is not None
        assert all(s.tag == "equal" for s in vm.expected_segments)
        assert any(
            s.tag == "produced_only" and "very" in s.text for s in vm.produced_segments
        )

    def test_removed_word_only_marked_on_expected_side(self):
        span = make_span(output="the sky is blue")

        vm = build_span_diff_view_model(span, "the sky is very blue")

        assert vm.produced_segments is not None
        assert vm.expected_segments is not None
        assert all(s.tag == "equal" for s in vm.produced_segments)
        assert any(
            s.tag == "expected_only" and "very" in s.text for s in vm.expected_segments
        )

    def test_segments_join_back_to_original_text(self):
        span = make_span(output="the sky is green today")
        expected = "the sky is blue today"

        vm = build_span_diff_view_model(span, expected)

        assert vm.expected_segments is not None
        assert vm.produced_segments is not None
        assert "".join(s.text for s in vm.expected_segments) == expected
        assert "".join(s.text for s in vm.produced_segments) == span.output


def make_citation_result(
    chunk_indices: list[int], supported: bool = True
) -> CitationVerificationResult:
    return CitationVerificationResult(
        claim_text="claim",
        chunk_indices=chunk_indices,
        supported=supported,
        reasoning="reasoning",
    )


class TestCitedChunkIndices:
    def test_empty_results_returns_empty_list(self):
        assert cited_chunk_indices([]) == []

    def test_returns_sorted_deduplicated_indices(self):
        results = [
            make_citation_result([3]),
            make_citation_result([1, 3]),
            make_citation_result([2]),
        ]

        assert cited_chunk_indices(results) == [1, 2, 3]

    def test_includes_unsupported_citations_indices_too(self):
        results = [make_citation_result([5], supported=False)]

        assert cited_chunk_indices(results) == [5]
