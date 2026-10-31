"""Tests for loading one bug and ranking stored defects."""

from pathlib import Path

import pytest

from jira_triage.domain.defect import Defect
from jira_triage.errors import JiraApiError, TriageError
from jira_triage.jira.lookup import JiraDefectLookup
from jira_triage.jira.mapper import DefectMapper
from jira_triage.jira.root_cause import DescriptionSectionParser
from jira_triage.rag.retrieval import DefectRetriever
from jira_triage.store.chroma_store import ChromaDefectVectorStore, StoredDefect


def _issue(key: str, summary: str, issue_type: str = "Bug", project: str = "FDE") -> dict:
    return {
        "key": key,
        "fields": {
            "summary": summary,
            "description": "Orders stay pending.\n\nRoot Cause: Secret rotated",
            "project": {"key": project},
            "priority": {"name": "High"},
            "issuetype": {"name": issue_type},
        },
    }


class FakeIssues:
    def __init__(self, issue: dict | None = None, error: Exception | None = None) -> None:
        self._issue = issue
        self._error = error
        self.requested: tuple[str, list[str]] | None = None

    def get_issue(self, issue_key: str, fields: list[str]) -> dict:
        self.requested = (issue_key, fields)
        if self._error:
            raise self._error
        assert self._issue is not None
        return self._issue


def _lookup(client: FakeIssues, project: str = "FDE", issue_type: str = "Bug") -> JiraDefectLookup:
    return JiraDefectLookup(
        client=client,
        mapper=DefectMapper(DescriptionSectionParser()),
        project_key=project,
        issue_type=issue_type,
    )


def test_lookup_maps_one_bug() -> None:
    client = FakeIssues(_issue("FDE-4", "Payment webhook silently failing"))

    defect = _lookup(client).get("fde-4")

    assert client.requested == (
        "FDE-4",
        ["summary", "description", "priority", "project", "issuetype"],
    )
    assert defect.defect_id == "FDE-4"
    assert defect.summary == "Payment webhook silently failing"
    assert defect.description == "Orders stay pending."
    assert defect.root_cause == "Secret rotated"


def test_lookup_reports_a_missing_issue() -> None:
    client = FakeIssues(error=JiraApiError(404, "Issue does not exist.", "/rest/api/3/issue/FDE-99"))

    with pytest.raises(TriageError, match="Jira has no issue FDE-99"):
        _lookup(client).get("FDE-99")


def test_lookup_rejects_a_different_issue_type() -> None:
    client = FakeIssues(_issue("FDE-4", "A story", issue_type="Story"))

    with pytest.raises(TriageError, match="FDE-4 is a Story"):
        _lookup(client).get("FDE-4")


def test_lookup_rejects_another_project() -> None:
    client = FakeIssues(_issue("PAY-4", "Other project", project="PAY"))

    with pytest.raises(TriageError, match="project FDE"):
        _lookup(client).get("PAY-4")


class FixedEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def _stored(defect_id: str, embedding: list[float], root_cause: str) -> StoredDefect:
    return StoredDefect(
        defect_id=defect_id,
        text=f"{defect_id} text",
        root_cause=root_cause,
        embedding=embedding,
    )


def test_nearest_excludes_the_queried_defect_and_ranks_by_cosine(tmp_path: Path) -> None:
    store = ChromaDefectVectorStore(tmp_path / "chroma")
    store.upsert(
        [
            _stored("FDE-3", [1.0, 0.0], "Same vector"),
            _stored("FDE-4", [0.6, 0.8], "Closer"),
            _stored("FDE-5", [0.0, 1.0], "Farther"),
        ]
    )

    neighbors = store.nearest([1.0, 0.0], limit=3, exclude_id="FDE-3")

    assert [neighbor.defect_id for neighbor in neighbors] == ["FDE-4", "FDE-5"]
    assert neighbors[0].distance < neighbors[1].distance
    assert neighbors[0].root_cause == "Closer"


def test_triage_returns_the_top_matches_without_the_entered_defect(tmp_path: Path) -> None:
    store = ChromaDefectVectorStore(tmp_path / "chroma")
    store.upsert(
        [
            _stored("FDE-3", [1.0, 0.0], "Same vector"),
            _stored("FDE-4", [0.8, 0.6], "Closer"),
            _stored("FDE-5", [0.6, 0.8], "Middle"),
            _stored("FDE-6", [0.0, 1.0], "Farther"),
        ]
    )
    entered = Defect(
        defect_id="FDE-3",
        summary="Inventory oversell",
        description="Stock went negative.",
        project_key="FDE",
        priority="Highest",
        logs="",
        root_cause="Isolation level",
    )

    class OneDefect:
        def get(self, defect_id: str) -> Defect:
            assert defect_id == "FDE-3"
            return entered

    result = DefectRetriever(OneDefect(), FixedEmbedder(), store, limit=3).retrieve("FDE-3")

    assert result.defect is entered
    assert [match.defect_id for match in result.matches] == ["FDE-4", "FDE-5", "FDE-6"]
    assert result.matches[0].similarity > result.matches[1].similarity > result.matches[2].similarity
    assert result.matches[0].document == "FDE-4 text"
