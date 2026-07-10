"""Streamlit entrypoint for the trace view.

Run with: streamlit run src/frontend/app.py

Opening a trace colors its nodes green/yellow purely from each `Span`'s
`confidence_score` — no LLM call. Root-cause coloring (red) requires an
explicit "Diagnose root cause" click, which runs `run_diagnosis` (real
Anthropic/OpenAI spend) and caches the result in `st.session_state` keyed by
trace_id, so repeat views of the same trace within a session don't re-spend.
"""

from __future__ import annotations

import streamlit as st

from src.config import settings
from src.frontend import detail_panel
from src.frontend.diagnosis_service import DiagnosisResult, run_diagnosis
from src.frontend.graph_render import render_graph
from src.frontend.view_models import (
    build_graph_view_model,
    root_cause_span_id_from_diagnosis,
)
from src.tracing.index import TraceRecord, init_trace_index, list_trace_records
from src.tracing.models import TraceStatus
from src.tracing.storage import load_trace

st.set_page_config(page_title="Trace View", layout="wide")
st.title("Pipeline Trace View")

init_trace_index(settings.sqlite_db_path)

_STATUS_OPTIONS: list[TraceStatus | None] = [None, "success", "degraded", "failure"]


def _format_record(record: TraceRecord) -> str:
    score = f"{record.final_score:.2f}" if record.final_score is not None else "—"
    return f"{record.timestamp:%Y-%m-%d %H:%M} · {record.status} · score={score}"


with st.sidebar:
    st.header("Traces")
    status_filter = st.selectbox(
        "Status filter",
        _STATUS_OPTIONS,
        format_func=lambda s: "all" if s is None else s,
    )
    records = list_trace_records(
        settings.sqlite_db_path, status=status_filter, limit=100
    )

    if not records:
        st.info("No persisted traces found yet.")
        st.stop()

    selected_record = st.selectbox("Trace", records, format_func=_format_record)

trace_id = selected_record.trace_id

try:
    trace = load_trace(trace_id, settings.trace_output_dir)
except FileNotFoundError:
    st.error(f"Trace file for {trace_id} is missing from {settings.trace_output_dir}.")
    st.stop()

diagnosis_cache_key = f"diagnosis::{trace_id}"

with st.sidebar:
    st.divider()
    if trace.status == "success":
        st.caption("Trace succeeded — no diagnosis needed.")
    elif diagnosis_cache_key in st.session_state:
        st.success("Root-cause diagnosis already run for this trace this session.")
    elif st.button("Diagnose root cause", key="diagnose_button"):
        with st.spinner("Running root-cause diagnosis (LLM calls)..."):
            st.session_state[diagnosis_cache_key] = run_diagnosis(trace, settings)
        st.rerun()

    diagnosis_result: DiagnosisResult | None = st.session_state.get(diagnosis_cache_key)
    if diagnosis_result is not None:
        if diagnosis_result.diagnosis is None:
            st.caption("Diagnosis ran: no unhealthy span found.")
        else:
            if diagnosis_result.category is not None:
                st.markdown(f"**Category:** {diagnosis_result.category.category}")
                st.caption(diagnosis_result.category.rationale)
            if diagnosis_result.evidence_chain is not None:
                with st.expander("Evidence chain narrative", expanded=True):
                    st.write(diagnosis_result.evidence_chain.narrative)

root_cause_span_id = root_cause_span_id_from_diagnosis(
    diagnosis_result.diagnosis if diagnosis_result is not None else None
)
view_model = build_graph_view_model(
    trace,
    root_cause_span_id=root_cause_span_id,
    low_confidence_threshold=settings.root_cause_quality_threshold,
)

col_graph, col_detail = st.columns([3, 2])

clicked_span_id: str | None = None
with col_graph:
    if not view_model.nodes:
        st.info("This trace has no spans.")
    else:
        clicked_span_id = render_graph(view_model)
        if clicked_span_id is not None:
            st.session_state["selected_span_id"] = clicked_span_id

selected_span_id = st.session_state.get("selected_span_id")
selected_span = next(
    (span for span in trace.spans if span.span_id == selected_span_id), None
)

with col_detail:
    if selected_span is not None:
        order = next(
            i for i, span in enumerate(trace.spans) if span.span_id == selected_span_id
        )
        status = next(
            node.status for node in view_model.nodes if node.span_id == selected_span_id
        )
        detail_panel.render(selected_span, status, order)
    else:
        st.caption("Click a node in the graph to see its details.")
