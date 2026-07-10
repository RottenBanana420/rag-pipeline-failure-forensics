"""Streamlit Flow adapter: renders a TraceGraphViewModel as an interactive
node graph and returns the clicked span_id (or None).

Written against `streamlit-flow-component` 1.6.1 (pinned via the `frontend`
extra in pyproject.toml, looked up via Context7 at implementation time) —
re-verify `StreamlitFlowNode`/`StreamlitFlowEdge`/`ManualLayout`/
`get_node_on_click` against the currently installed version if this stops
matching behavior, since component APIs can shift across releases.
"""

from __future__ import annotations

import streamlit as st
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowEdge, StreamlitFlowNode
from streamlit_flow.layouts import ManualLayout
from streamlit_flow.state import StreamlitFlowState

from src.frontend.view_models import (
    NODE_STATUS_COLOR,
    NodeViewModel,
    TraceGraphViewModel,
)

_NODE_X_SPACING = 220
_GATE_BORDER = "3px dashed #333333"
_NORMAL_BORDER = "1px solid #333333"


def _to_flow_node(node: NodeViewModel) -> StreamlitFlowNode:
    return StreamlitFlowNode(
        id=node.span_id,
        pos=(node.order * _NODE_X_SPACING, 0),
        data={"content": node.label},
        node_type="default",
        source_position="right",
        target_position="left",
        style={
            "background": NODE_STATUS_COLOR[node.status],
            "color": "white",
            "border": _GATE_BORDER if node.is_gate else _NORMAL_BORDER,
        },
    )


def _to_flow_edge(source: str, target: str) -> StreamlitFlowEdge:
    return StreamlitFlowEdge(id=f"{source}->{target}", source=source, target=target)


def render_graph(view_model: TraceGraphViewModel) -> str | None:
    """Render *view_model* as a flow diagram; return the clicked span_id, if any.

    Keyed by trace_id plus the current set of root-cause node ids, so that
    running an on-demand diagnosis (which recolors a node red from outside
    this function) mounts a fresh component with the new coloring already
    baked into its initial nodes, rather than needing to mutate an existing
    `StreamlitFlowState`'s node list in place.
    """
    root_cause_ids = tuple(
        n.span_id for n in view_model.nodes if n.status == "root_cause"
    )
    component_key = f"trace_flow::{view_model.trace_id}::{root_cause_ids}"
    # Deliberately distinct from component_key: passing the same string as
    # both a component's `key=` and our own st.session_state storage key
    # collides, since Streamlit auto-syncs a keyed component's raw return
    # value into st.session_state[key] itself, clobbering our own
    # StreamlitFlowState object with the raw frontend payload.
    storage_key = f"_flow_state::{component_key}"

    if storage_key not in st.session_state:
        nodes = [_to_flow_node(n) for n in view_model.nodes]
        edges = [_to_flow_edge(source, target) for source, target in view_model.edges]
        st.session_state[storage_key] = StreamlitFlowState(nodes, edges)

    st.session_state[storage_key] = streamlit_flow(
        component_key,
        st.session_state[storage_key],
        layout=ManualLayout(),
        fit_view=True,
        get_node_on_click=True,
        height=300,
    )

    selected_id: str | None = st.session_state[storage_key].selected_id
    return selected_id
