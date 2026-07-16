"""Streamlit page bodies for the multipage app (`src/frontend/app.py`).

`query_dashboard.py` is the end-user-facing Q&A page; `trace_view.py` is the
engineer-facing forensics page. Named `app_pages/` rather than the legacy
`pages/` to use Streamlit's modern `st.navigation`/`st.Page` API instead of
`pages/`'s old auto-discovery, which conflicts with it.
"""
