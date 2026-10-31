"""Pull defects from Jira Cloud."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol

from jira_triage.domain.defect import Defect
from jira_triage.domain.source import DefectSource
from jira_triage.jira.mapper import DefectMapper
from jira_triage.jira.query import BugQuery


class IssueSearcher(Protocol):
    """Searches Jira issues by JQL."""

    def iter_issues(self, jql: str, fields: list[str]) -> Iterator[dict[str, Any]]:
        """Yield matching issues."""


class JiraDefectSource(DefectSource):
    """Load every matching issue and map it to a Defect."""

    def __init__(self, client: IssueSearcher, query: BugQuery, mapper: DefectMapper) -> None:
        self._client = client
        self._query = query
        self._mapper = mapper

    def pull(self) -> list[Defect]:
        requested = ["summary", "description", "priority", "project"]
        return [
            self._mapper.map_issue(issue)
            for issue in self._client.iter_issues(self._query.jql(), requested)
        ]
