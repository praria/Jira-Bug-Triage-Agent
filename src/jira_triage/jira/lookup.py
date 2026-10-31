"""Load one defect from Jira by its issue key."""

from __future__ import annotations

from typing import Any, Protocol

from jira_triage.domain.defect import Defect
from jira_triage.errors import JiraApiError, TriageError
from jira_triage.jira.issue_key import require_defect_id
from jira_triage.jira.mapper import DefectMapper

_ISSUE_FIELDS = ["summary", "description", "priority", "project", "issuetype"]


class IssueReader(Protocol):
    """Reads one Jira issue by key."""

    def get_issue(self, issue_key: str, fields: list[str]) -> dict[str, Any]:
        """Return the issue payload."""


class JiraDefectLookup:
    """Fetch one bug in the configured project and map it to a Defect."""

    def __init__(
        self,
        client: IssueReader,
        mapper: DefectMapper,
        project_key: str,
        issue_type: str,
    ) -> None:
        self._client = client
        self._mapper = mapper
        self._project_key = project_key
        self._issue_type = issue_type

    def get(self, defect_id: str) -> Defect:
        key = require_defect_id(defect_id)
        try:
            issue = self._client.get_issue(key, _ISSUE_FIELDS)
        except JiraApiError as error:
            if error.status_code == 404:
                raise TriageError(f"Jira has no issue {key}.") from error
            raise
        fields = issue.get("fields") or {}
        found_type = _issue_type_name(fields)
        if found_type.casefold() != self._issue_type.casefold():
            label = found_type or "another issue type"
            raise TriageError(f"{key} is a {label}. Enter a {self._issue_type} key.")
        defect = self._mapper.map_issue(issue)
        if defect.project_key != self._project_key:
            raise TriageError(
                f"{key} is in project {defect.project_key}. This app triages project {self._project_key}."
            )
        return defect


def _issue_type_name(fields: dict[str, Any]) -> str:
    issue_type = fields.get("issuetype")
    if isinstance(issue_type, dict):
        return str(issue_type.get("name") or "").strip()
    return ""
