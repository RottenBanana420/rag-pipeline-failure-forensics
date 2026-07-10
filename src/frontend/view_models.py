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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from src.tracing.models import PipelineStep, Span, Trace

if TYPE_CHECKING:
    from src.analysis.root_cause import RootCauseDiagnosis

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
