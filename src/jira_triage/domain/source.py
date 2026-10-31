"""Source of defect records. Later steps depend on this, not on Jira."""

from abc import ABC, abstractmethod

from jira_triage.domain.defect import Defect


class DefectSource(ABC):
    """Loads the defects that will be indexed and compared."""

    @abstractmethod
    def pull(self) -> list[Defect]:
        """Return every defect in the configured scope."""
