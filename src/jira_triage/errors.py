"""Expected failures raised by the triage agent."""


class TriageError(Exception):
    """Base error for failures the caller can report directly."""


class ConfigurationError(TriageError):
    """Raised when required settings are missing or invalid."""


class JiraApiError(TriageError):
    """Raised when Jira Cloud returns an error response."""

    def __init__(self, status_code: int, message: str, path: str) -> None:
        self.status_code = status_code
        self.path = path
        super().__init__(f"Jira API {status_code} {path}: {message}")
