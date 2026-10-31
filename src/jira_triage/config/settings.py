"""Load Jira connection settings from the environment."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from jira_triage.errors import ConfigurationError

_REQUIRED = (
    "JIRA_BASE_URL",
    "JIRA_EMAIL",
    "JIRA_API_TOKEN",
    "JIRA_PROJECT_KEY",
)
_PROJECT_KEY = re.compile(r"^[A-Z][A-Z0-9]{1,9}$")


def require_project_key(value: str) -> str:
    """Return a normalized Jira project key, or raise if it is unsafe to put in JQL."""
    key = value.strip().upper()
    if _PROJECT_KEY.fullmatch(key) is None:
        raise ConfigurationError(
            "JIRA_PROJECT_KEY must be 2 to 10 letters or digits and start with a letter."
        )
    return key


@dataclass(frozen=True)
class Settings:
    """Connection settings for one Jira Cloud site and project."""

    jira_base_url: str
    jira_email: str
    jira_api_token: str = field(repr=False)
    jira_project_key: str
    root_cause_label: str = "Root Cause"
    jira_issue_type: str = "Bug"
    embedding_model: str = "all-MiniLM-L6-v2"
    chroma_path: str = ".chroma"
    gemini_api_key: str = field(default="", repr=False)
    gemini_model: str = "gemini-3.5-flash"

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> Settings:
        """Build settings from a .env file and the process environment.

        Values already set in the process environment win over the file.
        """
        if env_file is not None:
            load_dotenv(env_file, override=False)
        else:
            load_dotenv(override=False)

        missing = [name for name in _REQUIRED if not os.environ.get(name, "").strip()]
        if missing:
            joined = ", ".join(missing)
            raise ConfigurationError(
                f"Missing Jira settings: {joined}. Copy .env.example to .env and fill in the values."
            )

        base_url = os.environ["JIRA_BASE_URL"].strip().rstrip("/")
        if not base_url.startswith("https://"):
            raise ConfigurationError("JIRA_BASE_URL must start with https://")

        issue_type = os.environ.get("JIRA_ISSUE_TYPE", "Bug").strip() or "Bug"
        label = os.environ.get("JIRA_ROOT_CAUSE_LABEL", "").strip()
        if not label:
            label = os.environ.get("JIRA_ROOT_CAUSE_FIELD", "").strip() or "Root Cause"
        return cls(
            jira_base_url=base_url,
            jira_email=os.environ["JIRA_EMAIL"].strip(),
            jira_api_token=os.environ["JIRA_API_TOKEN"].strip(),
            jira_project_key=require_project_key(os.environ["JIRA_PROJECT_KEY"]),
            root_cause_label=label,
            jira_issue_type=issue_type,
            embedding_model=os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2").strip()
            or "all-MiniLM-L6-v2",
            chroma_path=os.environ.get("CHROMA_PATH", ".chroma").strip() or ".chroma",
            gemini_api_key=os.environ.get("GEMINI_API_KEY", "").strip(),
            gemini_model=os.environ.get("GEMINI_MODEL", "gemini-3.5-flash").strip() or "gemini-3.5-flash",
        )
