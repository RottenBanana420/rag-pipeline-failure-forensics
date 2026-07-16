"""Streamlit entrypoint for the RAG pipeline frontend.

Run with: streamlit run src/frontend/app.py

Two pages, via st.navigation/st.Page (app_pages/, not the legacy pages/
auto-discovery, per the project's Streamlit conventions):

- Query dashboard: end-user-facing Q&A over the ingested corpus.
- Trace view: engineer-facing forensics for any persisted Trace, including
  every question asked through the query dashboard.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="RAG pipeline", layout="wide")

page = st.navigation(
    [
        st.Page(
            "app_pages/query_dashboard.py",
            title="Query dashboard",
            icon=":material/chat:",
        ),
        st.Page(
            "app_pages/trace_view.py",
            title="Trace view",
            icon=":material/route:",
        ),
    ]
)
page.run()
