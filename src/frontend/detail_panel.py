"""Renders the node-detail panel for a selected Span."""

from __future__ import annotations

import streamlit as st

from src.frontend.view_models import NODE_STATUS_COLOR, NodeStatus
from src.tracing.models import Span

_EMBEDDINGS_NOTE = (
    "Embeddings not captured in this trace — `Span` has no embeddings field."
)
_EMBEDDING_ELIGIBLE_STEPS = ("retrieval", "ranking")


def render(span: Span, status: NodeStatus, order: int) -> None:
    """Render the full detail panel for *span*, given its derived *status*
    (green/yellow/red) and its 0-indexed *order* within the trace."""
    st.subheader(f"{order + 1}. {span.step}")

    badge_color = NODE_STATUS_COLOR[status]
    gate_badge = " &nbsp; `gate span`" if span.is_gate else ""
    st.markdown(
        f"<span style='background-color:{badge_color};color:white;"
        f"padding:2px 8px;border-radius:4px;'>{status}</span>{gate_badge}",
        unsafe_allow_html=True,
    )

    col_latency, col_tokens, col_confidence = st.columns(3)
    col_latency.metric("Latency (ms)", f"{span.latency_ms:.1f}")
    col_tokens.metric(
        "Tokens", span.token_count if span.token_count is not None else "—"
    )
    col_confidence.metric(
        "Confidence",
        span.confidence_score if span.confidence_score is not None else "—",
    )

    if span.error:
        st.error(span.error)

    st.text_area("Input", span.input, height=150, disabled=True)
    st.text_area("Output", span.output, height=150, disabled=True)

    if span.llm_prompt is not None:
        with st.expander("LLM Prompt"):
            st.text(span.llm_prompt)

    if span.step in _EMBEDDING_ELIGIBLE_STEPS:
        st.info(_EMBEDDINGS_NOTE)
