"""Pure, Streamlit-independent view-model construction for the trace view.

`node_status`/`build_graph_view_model` derive a `TraceGraphViewModel` from a
`Trace` plus an optional root-cause span id — no Streamlit or LLM import, so
this module is fully unit-testable on its own (mirrors `find_root_cause_span`
having no HTTP/UI concern of its own).

`Trace.spans` is a flat, execution-order list with no parent/child field, and
multiple spans can share the same `step` (e.g. `HybridRetriever`'s dense and
sparse legs are both `step="retrieval"`). So each span becomes its own node,
positioned/connected in `trace.spans` order — not grouped by step name.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Literal

from src.tracing.models import PipelineStep, Span, Trace

if TYPE_CHECKING:
    from src.analysis.root_cause import RootCauseDiagnosis
    from src.generation.citation_verifier import CitationVerificationResult

NodeStatus = Literal["healthy", "low_confidence", "root_cause"]

NODE_STATUS_COLOR: dict[NodeStatus, str] = {
    "healthy": "#2ecc71",
    "low_confidence": "#f1c40f",
    "root_cause": "#e74c3c",
}


def node_status(
    span: Span,
    *,
    root_cause_span_id: str | None,
    low_confidence_threshold: int,
) -> NodeStatus:
    """Classify *span* as healthy/low_confidence/root_cause for coloring.

    A root-cause match wins outright (red), regardless of `confidence_score`.
    Otherwise `confidence_score <= low_confidence_threshold` is yellow;
    `None` or above the threshold is green. `low_confidence_threshold` shares
    the same 1-5 scale/semantics as `settings.root_cause_quality_threshold`
    ("at or below this is unreasonable") but is passed explicitly rather than
    read from `Settings`, so this function stays testable with no env/config.
    """
    if root_cause_span_id is not None and span.span_id == root_cause_span_id:
        return "root_cause"
    if (
        span.confidence_score is not None
        and span.confidence_score <= low_confidence_threshold
    ):
        return "low_confidence"
    return "healthy"


@dataclass(frozen=True)
class NodeViewModel:
    span_id: str
    order: int
    step: PipelineStep
    label: str
    status: NodeStatus
    is_gate: bool


@dataclass(frozen=True)
class TraceGraphViewModel:
    trace_id: str
    nodes: list[NodeViewModel]
    edges: list[tuple[str, str]]


def build_graph_view_model(
    trace: Trace,
    *,
    root_cause_span_id: str | None,
    low_confidence_threshold: int,
) -> TraceGraphViewModel:
    """Build the graph view-model for *trace*, one node per span in order."""
    nodes = [
        NodeViewModel(
            span_id=span.span_id,
            order=i,
            step=span.step,
            label=f"{i + 1}. {span.step}",
            status=node_status(
                span,
                root_cause_span_id=root_cause_span_id,
                low_confidence_threshold=low_confidence_threshold,
            ),
            is_gate=span.is_gate,
        )
        for i, span in enumerate(trace.spans)
    ]
    edges = [(nodes[i].span_id, nodes[i + 1].span_id) for i in range(len(nodes) - 1)]
    return TraceGraphViewModel(trace_id=trace.trace_id, nodes=nodes, edges=edges)


def root_cause_span_id_from_diagnosis(
    diagnosis: RootCauseDiagnosis | None,
) -> str | None:
    """Extract the root-cause span's id from a diagnosis, if any."""
    return diagnosis.root_cause_span.span_id if diagnosis is not None else None


def cited_chunk_indices(
    citation_results: list[CitationVerificationResult],
) -> list[int]:
    """Sorted, deduplicated chunk indices cited across all citation results.

    Used to render one "jump to source" button per distinct cited chunk in
    the query dashboard, regardless of how many claims cite it.
    """
    return sorted(
        {index for result in citation_results for index in result.chunk_indices}
    )


_WHITESPACE_RE = re.compile(r"(\s+)")

DiffTag = Literal["equal", "expected_only", "produced_only"]


@dataclass(frozen=True)
class DiffSegment:
    text: str
    tag: DiffTag


@dataclass(frozen=True)
class SpanDiffViewModel:
    span_id: str
    received: str
    produced: str
    expected: str | None
    expected_segments: tuple[DiffSegment, ...] | None
    produced_segments: tuple[DiffSegment, ...] | None


def _tokenize(text: str) -> list[str]:
    """Split text into whitespace-preserving tokens for word-level diffing."""
    return [token for token in _WHITESPACE_RE.split(text) if token != ""]


def build_span_diff_view_model(
    span: Span, expected_output: str | None
) -> SpanDiffViewModel:
    """Build a side-by-side received/produced/expected diff for *span*.

    When *expected_output* is None (no human correction entered yet), both
    segment fields are None — there's nothing to diff against. Otherwise a
    word-level diff (same technique as `difflib.HtmlDiff`/`git diff
    --word-diff`) is computed between the expected output and `span.output`:
    `expected_segments` reflects the expected side (tagging text missing
    from what was produced), `produced_segments` reflects the produced side
    (tagging text not present in what was expected).
    """
    if expected_output is None:
        return SpanDiffViewModel(
            span_id=span.span_id,
            received=span.input,
            produced=span.output,
            expected=None,
            expected_segments=None,
            produced_segments=None,
        )

    expected_tokens = _tokenize(expected_output)
    produced_tokens = _tokenize(span.output)
    matcher = SequenceMatcher(None, expected_tokens, produced_tokens)
    opcodes = matcher.get_opcodes()

    expected_segments = [
        DiffSegment(
            text="".join(expected_tokens[i1:i2]),
            tag="equal" if tag == "equal" else "expected_only",
        )
        for tag, i1, i2, _j1, _j2 in opcodes
        if i1 != i2
    ]
    produced_segments = [
        DiffSegment(
            text="".join(produced_tokens[j1:j2]),
            tag="equal" if tag == "equal" else "produced_only",
        )
        for tag, _i1, _i2, j1, j2 in opcodes
        if j1 != j2
    ]

    return SpanDiffViewModel(
        span_id=span.span_id,
        received=span.input,
        produced=span.output,
        expected=expected_output,
        expected_segments=tuple(expected_segments),
        produced_segments=tuple(produced_segments),
    )
