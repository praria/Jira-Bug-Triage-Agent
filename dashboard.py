"""Streamlit app for defect embeddings and the Chroma database.

Run from the project root:

    streamlit run dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

st.set_page_config(page_title="Defect triage", layout="wide")

page = st.navigation(
    [
        st.Page(ROOT / "app_pages" / "triage.py", title="Triage", default=True),
        st.Page(ROOT / "app_pages" / "embeddings.py", title="Embeddings"),
        st.Page(ROOT / "app_pages" / "chroma_database.py", title="Chroma database"),
    ]
)
page.run()
