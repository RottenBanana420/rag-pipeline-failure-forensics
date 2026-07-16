"""End-user-facing query dashboard page.

Asking a question runs the full `ask_question` pipeline (real Anthropic/
OpenAI spend: one generation call, one citation-judge call per citation, one
completeness-judge call) — gated behind the form's submit button, and the
resulting `QueryResult` is cached in `st.session_state` keyed by the asked
query text so unrelated reruns (toggling the hybrid/dense-only comparison,
clicking a citation button) never re-spend. Every question is persisted as a
real `Trace`, inspectable afterward via "View full trace" — the query
dashboard is the first live (non-test) caller of `persist_trace`.

Citations are rendered as plain `[N]` markers in the answer text (Streamlit
has no in-page DOM anchor/scroll-to primitive), with a row of buttons below —
one per distinct cited chunk — that highlight the matching chunk card
further down the page. This is the pragmatic native-Streamlit reading of
"clickable citations linking to source chunk."

The hybrid-vs-dense-only comparison stays cheap: only an extra
`DenseRetriever.retrieve()` call for chunk-list display, no second
generation call or confidence score, and it runs outside the persisted
trace — a UI-only exploration aid, not part of the canonical request.
"""

from __future__ import annotations

import streamlit as st

from src.config import settings
from src.frontend.query_service import (
    QueryResult,
    RetrieverBundle,
    ask_question,
    build_hybrid_retriever,
)
from src.frontend.view_models import cited_chunk_indices
from src.retrieval.models import VectorStoreHit

st.title("Query dashboard")
st.caption(
    "Ask a question over the ingested corpus. Answers are grounded and cited "
    "against retrieved chunks — click **View full trace** to see the "
    "pipeline steps behind any answer."
)


@st.cache_resource
def _retriever_bundle() -> RetrieverBundle:
    return build_hybrid_retriever(settings)


def _render_chunks(
    hits: list[VectorStoreHit],
    supported_by_index: dict[int, bool],
    focused_index: int | None,
    key_prefix: str,
) -> None:
    if not hits:
        st.caption("No chunks retrieved.")
        return
    for i, hit in enumerate(hits, start=1):
        with st.container(border=True, key=f"{key_prefix}_chunk_{i}"):
            header = f"[{i}] {hit.title}"
            if hit.section_heading:
                header += f" — {hit.section_heading}"
            st.markdown(f"**{header}**")
            with st.container(horizontal=True):
                st.caption(f"Similarity: {hit.similarity:.2f}")
                supported = supported_by_index.get(i)
                if supported is True:
                    st.badge(
                        "Cited · supported", icon=":material/check:", color="green"
                    )
                elif supported is False:
                    st.badge(
                        "Cited · unsupported", icon=":material/close:", color="red"
                    )
                if i == focused_index:
                    st.badge("Jumped to", icon=":material/my_location:", color="blue")
            st.text(hit.text)


bundle = _retriever_bundle()

with st.form("ask_form"):
    query = st.text_area("Ask a question", height=100)
    compare = st.toggle("Compare hybrid vs. dense-only retrieval")
    submitted = st.form_submit_button("Ask", icon=":material/send:")

if submitted and query.strip():
    with st.spinner("Retrieving, generating, and scoring confidence (LLM calls)..."):
        new_result = ask_question(query, bundle.hybrid, settings)
    st.session_state["last_result"] = new_result
    st.session_state["last_query"] = query
    st.session_state.pop("dense_only_hits", None)
    st.session_state.pop("dense_only_query", None)
    st.session_state.pop("focused_chunk_index", None)

result: QueryResult | None = st.session_state.get("last_result")
query_asked: str | None = st.session_state.get("last_query")

if result is None or query_asked is None:
    st.info(
        "Ask a question to see the generated answer, retrieved chunks, and "
        "confidence breakdown."
    )
    st.stop()

st.subheader("Answer")
if result.fallback is not None:
    st.warning(result.fallback.message)
    st.caption(result.fallback.retrieved_summary)
    if result.fallback.documents_to_check:
        st.markdown("**Documents to check manually:**")
        for doc in result.fallback.documents_to_check:
            st.markdown(f"- {doc}")
else:
    st.markdown(result.answer_text)

    cited_indices = cited_chunk_indices(result.citation_results)
    if cited_indices:
        st.caption("Jump to source:")
        with st.container(horizontal=True):
            for idx in cited_indices:
                if st.button(f"[{idx}]", key=f"cite_{idx}"):
                    st.session_state["focused_chunk_index"] = idx

st.subheader("Confidence")
confidence = result.confidence
with st.container(horizontal=True):
    st.metric("Retrieval", f"{confidence.retrieval_confidence:.0%}", border=True)
    st.metric("Citation coverage", f"{confidence.citation_coverage:.0%}", border=True)
    st.metric("Completeness", f"{confidence.answer_completeness:.0%}", border=True)
    st.metric("Composite", f"{confidence.composite:.0%}", border=True)

if st.button("View full trace", icon=":material/route:"):
    st.session_state["preselect_trace_id"] = result.trace_id
    st.switch_page("app_pages/trace_view.py")

st.divider()

supported_by_index: dict[int, bool] = {}
for citation_result in result.citation_results:
    for idx in citation_result.chunk_indices:
        supported_by_index[idx] = (
            supported_by_index.get(idx, False) or citation_result.supported
        )

focused_index = st.session_state.get("focused_chunk_index")

if compare:
    if (
        "dense_only_hits" not in st.session_state
        or st.session_state.get("dense_only_query") != query_asked
    ):
        with st.spinner("Retrieving dense-only comparison..."):
            st.session_state["dense_only_hits"] = bundle.dense.retrieve(
                query_asked, k=len(result.hits) or 5
            )
            st.session_state["dense_only_query"] = query_asked

    col_hybrid, col_dense = st.columns(2)
    with col_hybrid:
        st.subheader("Hybrid retrieval")
        _render_chunks(result.hits, supported_by_index, focused_index, "hybrid")
    with col_dense:
        st.subheader("Dense-only retrieval")
        _render_chunks(st.session_state["dense_only_hits"], {}, None, "dense")
else:
    st.subheader("Retrieved chunks")
    _render_chunks(result.hits, supported_by_index, focused_index, "hybrid")
