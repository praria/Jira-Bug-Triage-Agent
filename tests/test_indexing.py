"""Tests for embedding text and storing vectors."""

from pathlib import Path

from jira_triage.domain.defect import Defect
from jira_triage.domain.source import DefectSource
from jira_triage.rag.embeddings import DefectEmbeddingText
from jira_triage.indexing import DefectIndexer
from jira_triage.store.chroma_store import ChromaDefectVectorStore


def _defect(defect_id: str, summary: str, description: str, root_cause: str) -> Defect:
    return Defect(
        defect_id=defect_id,
        summary=summary,
        description=description,
        project_key="FDE",
        priority="Highest",
        logs="ERROR sample",
        root_cause=root_cause,
    )


class FixedSource(DefectSource):
    def __init__(self, defects: list[Defect]) -> None:
        self._defects = defects

    def pull(self) -> list[Defect]:
        return list(self._defects)


class FixedEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]


def test_embedding_text_joins_summary_and_description() -> None:
    defect = _defect("FDE-3", "Inventory oversell", "Stock went negative.", "Isolation level")

    assert DefectEmbeddingText().render(defect) == "Inventory oversell\n\nStock went negative."


def test_indexer_stores_defect_id_and_root_cause(tmp_path: Path) -> None:
    defects = [
        _defect("FDE-3", "Inventory oversell", "Stock went negative.", "Isolation level"),
        _defect("FDE-4", "Webhook failing", "Orders stay pending.", "Secret rotated"),
    ]
    store = ChromaDefectVectorStore(tmp_path / "chroma")

    count = DefectIndexer(FixedSource(defects), FixedEmbedder(), store).index()

    assert count == 2
    assert store.count() == 2
    stored = store.fetch("FDE-3")
    assert stored is not None
    assert stored.defect_id == "FDE-3"
    assert stored.root_cause == "Isolation level"
    assert stored.text == "Inventory oversell\n\nStock went negative."
    assert stored.embedding == [float(len(stored.text)), 1.0]


def test_reindex_replaces_the_same_defect(tmp_path: Path) -> None:
    store = ChromaDefectVectorStore(tmp_path / "chroma")
    first = _defect("FDE-3", "Old summary", "Old description.", "Old cause")
    updated = _defect("FDE-3", "New summary", "New description.", "New cause")

    DefectIndexer(FixedSource([first]), FixedEmbedder(), store).index()
    DefectIndexer(FixedSource([updated]), FixedEmbedder(), store).index()

    assert store.count() == 1
    stored = store.fetch("FDE-3")
    assert stored is not None
    assert stored.root_cause == "New cause"
    assert stored.text == "New summary\n\nNew description."


def test_list_all_returns_every_stored_defect_in_id_order(tmp_path: Path) -> None:
    store = ChromaDefectVectorStore(tmp_path / "chroma")
    defects = [
        _defect("FDE-10", "Later", "Tenth.", "Cause 10"),
        _defect("FDE-3", "Earlier", "Third.", "Cause 3"),
    ]
    DefectIndexer(FixedSource(defects), FixedEmbedder(), store).index()

    listed = store.list_all()

    assert [record.defect_id for record in listed] == ["FDE-3", "FDE-10"]
