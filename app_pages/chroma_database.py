"""Monitor the local Chroma database."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st

from jira_triage.dashboard.chroma_monitor import ChromaDatabaseMonitor, DatabaseReport


def _disk(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def _status_label(report: DatabaseReport) -> str:
    if not report.reachable:
        return "Unavailable"
    if report.issues:
        return "Attention"
    if report.record_count == 0:
        return "Empty"
    return "Healthy"


st.title("Chroma database")
st.caption("Connection, collection, and embedding health for the local vector database.")

if st.button("Refresh"):
    st.rerun()

report = ChromaDatabaseMonitor(ROOT / ".chroma").status()
label = _status_label(report)

status_column, collections_column, records_column, disk_column = st.columns(4)
status_column.metric("Status", label)
collections_column.metric("Collections", report.collection_count)
records_column.metric("Records", report.record_count)
disk_column.metric("Disk", _disk(report.disk_bytes))

st.write(f"Path: `{report.path}`")
if report.reachable:
    st.write(f"Tenant: `{report.tenant}`  ·  Database: `{report.database}`")

if not report.reachable:
    st.error(report.message)
    st.stop()

if report.issues:
    for issue in report.issues:
        st.warning(issue)
else:
    st.success("The database responded and every stored embedding passed the checks.")

st.subheader("Collections")
st.dataframe(
    pd.DataFrame(
        [
            {
                "collection": collection.name,
                "records": collection.record_count,
                "dimensions": collection.dimensions,
                "distance": collection.distance,
                "metadata_fields": ", ".join(collection.metadata_fields),
                "empty_root_cause": collection.empty_root_cause,
                "empty_documents": collection.empty_documents,
                "search_ms": round(collection.search_ms, 2),
                "search_ok": collection.search_ok,
            }
            for collection in report.collections
        ]
    ),
    hide_index=True,
    width="stretch",
)

st.subheader("Records")
st.caption("defect_id and root_cause are metadata. The document is the summary and description that were embedded.")
records = [record for collection in report.collections for record in collection.records]
st.dataframe(
    pd.DataFrame(
        [
            {
                "defect_id": record.defect_id,
                "root_cause": record.root_cause,
                "document": record.document,
                "dimensions": record.dimensions,
                "vector_norm": round(record.vector_norm, 4),
            }
            for record in records
        ]
    ),
    hide_index=True,
    width="stretch",
    column_config={
        "defect_id": st.column_config.TextColumn("metadata.defect_id"),
        "root_cause": st.column_config.TextColumn("metadata.root_cause", width="large"),
        "document": st.column_config.TextColumn("document", width="large"),
    },
)

if records:
    selected_id = st.selectbox("Record", [record.defect_id or record.chroma_id for record in records])
    selected = next(record for record in records if (record.defect_id or record.chroma_id) == selected_id)
    st.subheader(selected.defect_id or selected.chroma_id)
    metadata_column, document_column = st.columns(2)
    metadata_column.markdown("**Metadata**")
    metadata_column.json({"defect_id": selected.defect_id, "root_cause": selected.root_cause})
    document_column.markdown("**Document**")
    document_column.text(selected.document)
