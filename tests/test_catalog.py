"""Tests for dashboard rows built from stored embeddings."""

from jira_triage.dashboard.catalog import EmbeddingCatalog, EmbeddingProjector, EmbeddingRow, closest_bugs
from jira_triage.store.chroma_store import StoredDefect


class MemoryStore:
    def __init__(self, records: list[StoredDefect]) -> None:
        self._records = records

    def upsert(self, records: list[StoredDefect]) -> None:
        self._records = list(records)

    def count(self) -> int:
        return len(self._records)

    def fetch(self, defect_id: str) -> StoredDefect | None:
        for record in self._records:
            if record.defect_id == defect_id:
                return record
        return None

    def list_all(self) -> list[StoredDefect]:
        return list(self._records)


def test_catalog_splits_summary_and_places_points() -> None:
    store = MemoryStore(
        [
            StoredDefect("FDE-3", "Inventory oversell\n\nStock went negative.", "Isolation", [1.0, 0.0, 0.0]),
            StoredDefect("FDE-4", "Webhook failing\n\nOrders stay pending.", "Secret", [0.0, 1.0, 0.0]),
        ]
    )

    rows = EmbeddingCatalog(store, EmbeddingProjector()).rows()

    assert [row.defect_id for row in rows] == ["FDE-3", "FDE-4"]
    assert rows[0].summary == "Inventory oversell"
    assert rows[0].description == "Stock went negative."
    assert rows[0].root_cause == "Isolation"
    assert rows[0].dimensions == 3
    assert rows[0].x != rows[1].x or rows[0].y != rows[1].y


def test_closest_bugs_ranks_the_more_similar_wording_first() -> None:
    selected = EmbeddingRow("FDE-3", "Oversell", "", "Isolation", 2, 0.0, 0.0, [1.0, 0.0])
    rows = [
        selected,
        EmbeddingRow("FDE-4", "Webhook", "", "Secret", 2, 0.0, 0.0, [0.6, 0.8]),
        EmbeddingRow("FDE-5", "Tax", "", "Outage", 2, 0.0, 0.0, [0.0, 1.0]),
    ]

    nearest = closest_bugs(selected, rows, limit=5)

    assert [bug.defect_id for bug in nearest] == ["FDE-4", "FDE-5"]
    assert nearest[0].similarity == 0.6
    assert nearest[0].summary == "Webhook"
