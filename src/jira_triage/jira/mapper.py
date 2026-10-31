"""Map a Jira issue payload onto a Defect."""

from __future__ import annotations

from typing import Any

from jira_triage.domain.defect import Defect
from jira_triage.jira.root_cause import DescriptionSectionParser


class DefectMapper:
    """Build a Defect from one Jira issue."""

    def __init__(self, section_parser: DescriptionSectionParser) -> None:
        self._section_parser = section_parser

    def map_issue(self, issue: dict[str, Any]) -> Defect:
        key = str(issue.get("key") or "").strip()
        if not key:
            raise ValueError("Jira issue is missing a key.")
        fields = issue.get("fields") or {}
        parts = self._section_parser.split(fields.get("description"))
        return Defect(
            defect_id=key,
            summary=str(fields.get("summary") or "").strip(),
            description=parts.description,
            project_key=_project_key(fields, key),
            priority=_priority(fields),
            logs=parts.logs,
            root_cause=parts.root_cause,
        )


def _project_key(fields: dict[str, Any], issue_key: str) -> str:
    project = fields.get("project")
    if isinstance(project, dict) and str(project.get("key") or "").strip():
        return str(project["key"]).strip()
    if "-" in issue_key:
        return issue_key.split("-", 1)[0]
    return ""


def _priority(fields: dict[str, Any]) -> str:
    priority = fields.get("priority")
    if isinstance(priority, dict):
        return str(priority.get("name") or "").strip()
    return ""
