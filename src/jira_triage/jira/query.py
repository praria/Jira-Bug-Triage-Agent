"""Build the JQL used to pull defects."""

from dataclasses import dataclass


def _jql_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


@dataclass(frozen=True)
class BugQuery:
    """JQL for every issue of one type in one project."""

    project_key: str
    issue_type: str = "Bug"

    def jql(self) -> str:
        return (
            f"project = {self.project_key} "
            f"AND issuetype = {_jql_string(self.issue_type)} "
            "ORDER BY key ASC"
        )
