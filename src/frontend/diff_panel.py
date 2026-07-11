"""Renders the diff view for a selected Span in a failed/degraded trace.

Side-by-side "received / produced / should have produced" comparison per
`docs/PROJECT_SPEC.md` (Phase 5, item 2). "Should have produced" comes from
a human-entered correction (`src.frontend.corrections`), not a golden
dataset — no golden-dataset system exists yet (Phase 6 placeholder).
"""

from __future__ import annotations

import html

import streamlit as st

from src.config import Settings
from src.frontend.corrections import load_correction, save_correction
from src.frontend.view_models import DiffSegment, build_span_diff_view_model
from src.tracing.models import Span

_SEGMENT_STYLE = {
    "equal": "",
    "expected_only": "background-color:#e74c3c33;text-decoration:line-through;",
    "produced_only": "background-color:#2ecc7133;font-weight:600;",
}


def _render_segments(segments: tuple[DiffSegment, ...]) -> str:
    parts = []
    for segment in segments:
        escaped = html.escape(segment.text)
        style = _SEGMENT_STYLE[segment.tag]
        if style:
            parts.append(f"<span style='{style}'>{escaped}</span>")
        else:
            parts.append(escaped)
    # white-space:pre-wrap preserves runs of spaces/newlines from the diffed
    # text verbatim (default HTML rendering would collapse them, hiding
    # exactly the kind of whitespace-only divergence this view exists to show).
    return f"<div style='white-space:pre-wrap'>{''.join(parts)}</div>"


def render(span: Span, trace_id: str, settings: Settings) -> None:
    """Render the diff view for *span* within *trace_id*."""
    st.subheader("Diff view")

    existing = load_correction(trace_id, span.span_id, settings.human_corrections_dir)
    widget_key = f"expected_output::{trace_id}::{span.span_id}"
    expected_input = st.text_area(
        "Expected output (human correction)",
        value=existing or "",
        key=widget_key,
        height=100,
    )
    if st.button("Save correction", key=f"save_correction::{widget_key}"):
        save_correction(
            trace_id, span.span_id, expected_input, settings.human_corrections_dir
        )
        st.toast("Correction saved.", icon="✅")

    # Diffs against the live textbox value, not the on-disk `existing` value,
    # so the highlighted divergence updates immediately as the user types/
    # blurs — it doesn't require a Save round-trip first. An empty box means
    # "no correction entered," same as an unset correction, so it's treated
    # as None rather than diffed against "".
    view_model = build_span_diff_view_model(span, expected_input or None)

    col_received, col_produced, col_expected = st.columns(3)
    with col_received:
        st.markdown("**Received**")
        st.text(view_model.received)
    with col_produced:
        st.markdown("**Produced**")
        if view_model.produced_segments is not None:
            st.html(_render_segments(view_model.produced_segments))
        else:
            st.text(view_model.produced)
    with col_expected:
        st.markdown("**Should have produced**")
        if view_model.expected_segments is not None:
            st.html(_render_segments(view_model.expected_segments))
        else:
            st.caption("Enter an expected output above to see the divergence.")
