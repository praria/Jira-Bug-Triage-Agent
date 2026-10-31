"""Tests for the object that pulls and maps defects."""

from jira_triage.jira.defect_source import JiraDefectSource
from jira_triage.jira.mapper import DefectMapper
from jira_triage.jira.query import BugQuery
from jira_triage.jira.root_cause import DescriptionSectionParser


class FakeJira:
    def __init__(self) -> None:
        self.requested_fields: list[str] | None = None
        self.jql: str | None = None

    def iter_issues(self, jql: str, fields: list[str]):
        self.jql = jql
        self.requested_fields = fields
        yield {
            "key": "PAY-7",
            "fields": {
                "summary": "Invoice tax is wrong",
                "project": {"key": "PAY"},
                "priority": {"name": "High"},
                "description": "\n".join(
                    [
                        "Tax rounds down.",
                        "",
                        "Logs: WARN [Tax] - rounded down",
                        "Root Cause: Rounding",
                    ]
                ),
            },
        }


def test_pull_maps_sections_priority_and_project() -> None:
    jira = FakeJira()
    source = JiraDefectSource(
        client=jira,
        query=BugQuery("PAY", "Bug"),
        mapper=DefectMapper(DescriptionSectionParser()),
    )

    defects = source.pull()

    assert jira.jql == 'project = PAY AND issuetype = "Bug" ORDER BY key ASC'
    assert jira.requested_fields == ["summary", "description", "priority", "project"]
    assert defects[0].to_dict() == {
        "summary": "Invoice tax is wrong",
        "description": "Tax rounds down.",
        "project_key": "PAY",
        "priority": "High",
        "labels": [
            "logs: WARN [Tax] - rounded down",
            "root_cause: Rounding",
        ],
    }


def test_issue_type_quotes_are_escaped() -> None:
    assert BugQuery("PAY", 'Bug "prod"').jql() == (
        'project = PAY AND issuetype = "Bug \\"prod\\"" ORDER BY key ASC'
    )
