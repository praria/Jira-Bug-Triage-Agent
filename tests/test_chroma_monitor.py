"""Tests for the Chroma database health report."""

from pathlib import Path

from jira_triage.dashboard.chroma_monitor import ChromaDatabaseMonitor
from jira_triage.store.chroma_store import ChromaDefectVectorStore, StoredDefect


def test_missing_database_is_unavailable(tmp_path: Path) -> None:
    path = tmp_path / "missing"

    report = ChromaDatabaseMonitor(path).status()

    assert report.reachable is False
    assert report.issues == ("The Chroma directory does not exist.",)
    assert not path.exists()


def test_reports_collection_counts_and_a_working_search(tmp_path: Path) -> None:
    path = tmp_path / "chroma"
    ChromaDefectVectorStore(path).upsert(
        [
            StoredDefect("FDE-3", "Inventory oversell\n\nStock went negative.", "Isolation", [1.0, 0.0]),
            StoredDefect("FDE-4", "Webhook failing\n\nOrders stay pending.", "", [0.0, 1.0]),
        ]
    )

    report = ChromaDatabaseMonitor(path).status()

    assert report.reachable is True
    assert report.heartbeat > 0
    assert report.disk_bytes > 0
    assert report.collection_count == 1
    collection = report.collections[0]
    assert collection.name == "defects"
    assert collection.record_count == 2
    assert collection.dimensions == 2
    assert collection.distance == "cosine"
    assert collection.metadata_fields == ("defect_id", "root_cause")
    assert collection.empty_root_cause == 1
    assert collection.empty_documents == 0
    assert collection.search_ok is True
    assert [record.defect_id for record in collection.records] == ["FDE-3", "FDE-4"]
    assert collection.records[0].document == "Inventory oversell\n\nStock went negative."
    assert collection.empty_defect_id == 0
    assert any("empty root cause" in issue for issue in report.issues)
