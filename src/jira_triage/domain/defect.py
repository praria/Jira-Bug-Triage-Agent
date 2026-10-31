"""A defect record independent of Jira's JSON shape."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Defect:
    """One defect pulled from Jira.

    ``description`` is the narrative text. ``logs`` and ``root_cause`` come
    from the Logs and Root Cause sections of that description. ``to_dict``
    writes those two sections into ``labels``.
    """

    defect_id: str
    summary: str
    description: str
    project_key: str
    priority: str
    logs: str
    root_cause: str

    @property
    def labels(self) -> list[str]:
        items: list[str] = []
        if self.logs:
            items.append(f"logs: {self.logs}")
        if self.root_cause:
            items.append(f"root_cause: {self.root_cause}")
        return items

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "description": self.description,
            "project_key": self.project_key,
            "priority": self.priority,
            "labels": self.labels,
        }
