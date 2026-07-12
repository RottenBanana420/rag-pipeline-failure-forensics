"""Streamlit entrypoint for the trace view.

Run with: streamlit run src/frontend/app.py

Opening a trace colors its nodes green/yellow purely from each `Span`'s
`confidence_score` — no LLM call. Root-cause coloring (red) requires an
explicit "Flag as bad output" click, which runs `run_diagnosis` (real
Anthropic/OpenAI spend) and caches the result in `st.session_state` keyed by
trace_id, so repeat views of the same trace within a session don't re-spend.
Flagging is available on every trace, including ones the pipeline itself
scored "success" — a human can still catch a bad output. Confirming or
overriding the diagnosis persists a `FlagRecord` (`src.frontend.flags`) only
on that explicit action; running the diagnosis alone writes nothing to disk.
Revisiting an already-flagged trace in a fresh session (no session-state
diagnosis) still colors the root-cause node red, from the persisted flag.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import get_args

import streamlit as st

from src.analysis.failure_categorizer import FailureCategory
from src.config import settings
from src.frontend import detail_panel, diff_panel
from src.frontend.diagnosis_service import DiagnosisResult, run_diagnosis
from src.frontend.flags import (
    FlagRecord,
    HumanReview,
    diagnosis_summary_from_result,
    load_flag,
    save_flag,
)
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


def _format_flagged_at(iso_timestamp: str) -> str:
    return datetime.fromisoformat(iso_timestamp).strftime("%Y-%m-%d %H:%M")


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
review_open_key = f"flag_review_open::{trace_id}"

diagnosis_result: DiagnosisResult | None = None

with st.sidebar:
    st.divider()
    st.subheader("Flag as bad output")

    flag_record = load_flag(trace_id, settings.flagged_traces_dir)
    review_open = st.session_state.setdefault(review_open_key, flag_record is None)

    if flag_record is not None and not review_open:
        verdict = "confirmed" if flag_record.human_review.confirmed else "overridden"
        st.success(
            f"Flagged {_format_flagged_at(flag_record.flagged_at)} — {verdict} "
            f"({flag_record.human_review.category})"
        )
        if flag_record.human_review.note:
            st.caption(f'Note: "{flag_record.human_review.note}"')
        if st.button("Redo review", icon=":material/flag:", key="redo_review_button"):
            st.session_state[review_open_key] = True
            st.rerun()
    else:
        if diagnosis_cache_key in st.session_state:
            st.success("Root-cause diagnosis already run for this trace this session.")
        elif st.button("Flag as bad output", icon=":material/flag:", key="flag_button"):
            with st.spinner("Running root-cause diagnosis (LLM calls)..."):
                st.session_state[diagnosis_cache_key] = run_diagnosis(trace, settings)
            st.rerun()

        diagnosis_result = st.session_state.get(diagnosis_cache_key)

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

            summary = diagnosis_summary_from_result(diagnosis_result)

            if summary is not None and st.button(
                "Confirm diagnosis", icon=":material/check:", key="confirm_flag_button"
            ):
                record = FlagRecord(
                    flagged_at=datetime.now(UTC).isoformat(),
                    diagnosis=summary,
                    human_review=HumanReview(
                        confirmed=True,
                        span_id=summary.root_cause_span_id,
                        category=summary.category,
                        note="",
                    ),
                )
                save_flag(trace_id, record, settings.flagged_traces_dir)
                st.session_state[review_open_key] = False
                st.toast("Diagnosis confirmed.", icon=":material/check:")
                st.rerun()

            # Always available once a diagnosis run has happened this
            # session — including the no-root-cause-found case (summary is
            # None), where this form is the only finalize path.
            if not trace.spans:
                st.caption("This trace has no spans to select.")
            else:
                with st.form(f"override_form::{trace_id}"):
                    st.caption("Override diagnosis")
                    default_index = next(
                        (
                            i
                            for i, span in enumerate(trace.spans)
                            if summary is not None
                            and span.span_id == summary.root_cause_span_id
                        ),
                        0,
                    )
                    override_span = st.selectbox(
                        "Root-cause span",
                        trace.spans,
                        index=default_index,
                        format_func=lambda s: f"{s.step} ({s.span_id[:8]})",
                    )
                    override_category = st.selectbox(
                        "Failure category", get_args(FailureCategory)
                    )
                    override_note = st.text_area("Note")
                    override_submitted = st.form_submit_button(
                        "Save override", icon=":material/edit:"
                    )

                if override_submitted:
                    record = FlagRecord(
                        flagged_at=datetime.now(UTC).isoformat(),
                        diagnosis=summary,
                        human_review=HumanReview(
                            confirmed=False,
                            span_id=override_span.span_id,
                            category=override_category,
                            note=override_note,
                        ),
                    )
                    save_flag(trace_id, record, settings.flagged_traces_dir)
                    st.session_state[review_open_key] = False
                    st.toast("Override saved.", icon=":material/edit:")
                    st.rerun()

root_cause_span_id = root_cause_span_id_from_diagnosis(
    diagnosis_result.diagnosis if diagnosis_result is not None else None
)
if (
    root_cause_span_id is None
    and flag_record is not None
    and flag_record.diagnosis is not None
):
    root_cause_span_id = flag_record.diagnosis.root_cause_span_id

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
        if trace.status != "success":
            st.divider()
            diff_panel.render(selected_span, trace_id, settings)
    else:
        st.caption("Click a node in the graph to see its details.")
