"""Embedding map and per-defect vectors."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st

import altair as alt

from jira_triage.dashboard.catalog import EmbeddingCatalog, EmbeddingRow, closest_bugs
from jira_triage.store.chroma_store import ChromaDefectVectorStore

st.title("Defect embeddings")
st.caption("Each point is one Jira bug. The vector is built from the summary and description.")


def _load_rows() -> list[EmbeddingRow]:
    store = ChromaDefectVectorStore(ROOT / ".chroma")
    return EmbeddingCatalog(store).rows()


rows = _load_rows()
if not rows:
    st.info("No embeddings are stored yet. Run `python index_defects.py` first.")
    st.stop()

dimensions = rows[0].dimensions
left, right = st.columns(2)
left.metric("Defects", len(rows))
right.metric("Vector size", dimensions)

chart = pd.DataFrame(
    {
        "defect_id": [row.defect_id for row in rows],
        "summary": [row.summary for row in rows],
        "x": [row.x for row in rows],
        "y": [row.y for row in rows],
    }
)
st.subheader("Embedding map")
st.scatter_chart(chart, x="x", y="y", color="defect_id", height=420)

table = pd.DataFrame(
    {
        "defect_id": [row.defect_id for row in rows],
        "summary": [row.summary for row in rows],
        "root_cause": [row.root_cause for row in rows],
        "dimensions": [row.dimensions for row in rows],
    }
)
st.subheader("Indexed defects")
st.dataframe(table, hide_index=True, width="stretch")

by_id = {row.defect_id: row for row in rows}
selected_id = st.selectbox("Bug", list(by_id), index=0)
selected = by_id[selected_id]
st.subheader(selected.defect_id)
st.write(selected.summary)
if selected.description:
    st.write(selected.description)
st.write(f"Root cause: {selected.root_cause}")

st.subheader("Closest bugs")
st.caption(
    "Bugs whose summary and description are most like the one you selected. "
    "100% would mean the same wording. The Triage page uses this same comparison."
)
nearest = closest_bugs(selected, rows, limit=5)
if not nearest:
    st.info("This is the only stored bug, so there is nothing to compare.")
else:
    comparison = pd.DataFrame(
        {
            "Bug": [bug.defect_id for bug in nearest],
            "Similarity": [round(bug.similarity * 100, 1) for bug in nearest],
            "Summary": [bug.summary for bug in nearest],
        }
    )
    chart = (
        alt.Chart(comparison)
        .mark_bar()
        .encode(
            x=alt.X("Bug:N", sort="-y", title="Bug"),
            y=alt.Y("Similarity:Q", scale=alt.Scale(domain=[0, 100]), title="Similarity (%)"),
            tooltip=["Bug", "Summary", alt.Tooltip("Similarity:Q", format=".1f", title="Similarity (%)")],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, use_container_width=True)
    st.dataframe(
        comparison.assign(Similarity=comparison["Similarity"].map(lambda value: f"{value:.1f}%")),
        hide_index=True,
        width="stretch",
    )

with st.expander(f"The {selected.dimensions} numbers behind this comparison"):
    st.caption(
        f"Each bug is stored as {selected.dimensions} numbers. "
        "Similar bugs have similar numbers. You do not need to read them one by one."
    )
    st.dataframe(
        pd.DataFrame(
            {"dimension": range(len(selected.embedding)), "value": selected.embedding}
        ),
        hide_index=True,
        width="stretch",
    )
