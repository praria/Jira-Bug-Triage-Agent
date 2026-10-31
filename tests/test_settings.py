"""Tests for environment loading."""

from pathlib import Path

import pytest

from jira_triage.config.settings import Settings
from jira_triage.errors import ConfigurationError


_KEYS = (
    "JIRA_BASE_URL",
    "JIRA_EMAIL",
    "JIRA_API_TOKEN",
    "JIRA_PROJECT_KEY",
    "JIRA_ROOT_CAUSE_FIELD",
    "JIRA_ISSUE_TYPE",
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _KEYS:
        monkeypatch.delenv(key, raising=False)


def test_from_env_reads_file(clean_env: None, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "JIRA_BASE_URL=https://acme.atlassian.net/",
                "JIRA_EMAIL=analyst@acme.com",
                "JIRA_API_TOKEN=secret-token",
                "JIRA_PROJECT_KEY=pay",
                "JIRA_ROOT_CAUSE_FIELD=Root Cause",
            ]
        ),
        encoding="utf-8",
    )

    settings = Settings.from_env(env_file)

    assert settings.jira_base_url == "https://acme.atlassian.net"
    assert settings.jira_project_key == "PAY"
    assert settings.jira_issue_type == "Bug"
    assert settings.root_cause_label == "Root Cause"
    assert settings.gemini_api_key == ""
    assert settings.gemini_model == "gemini-3.5-flash"
    assert "secret-token" not in repr(settings)


def test_missing_values_name_every_gap(clean_env: None, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("JIRA_BASE_URL=https://acme.atlassian.net\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="JIRA_API_TOKEN"):
        Settings.from_env(env_file)


def test_rejects_non_https_site(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JIRA_BASE_URL", "http://acme.atlassian.net")
    monkeypatch.setenv("JIRA_EMAIL", "analyst@acme.com")
    monkeypatch.setenv("JIRA_API_TOKEN", "secret-token")
    monkeypatch.setenv("JIRA_PROJECT_KEY", "PAY")
    monkeypatch.setenv("JIRA_ROOT_CAUSE_FIELD", "Root Cause")

    with pytest.raises(ConfigurationError, match="https://"):
        Settings.from_env()
