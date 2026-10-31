"""Tests for Jira search pagination and error handling."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from jira_triage.config.settings import Settings
from jira_triage.errors import JiraApiError
from jira_triage.jira.client import JiraApiRootResolver, JiraCloudClient


def _settings() -> Settings:
    return Settings(
        jira_base_url="https://example.atlassian.net",
        jira_email="analyst@example.com",
        jira_api_token="secret-token",
        jira_project_key="PAY",
        root_cause_label="Root Cause",
    )


def _response(status: int, payload: Any) -> MagicMock:
    response = MagicMock()
    response.status_code = status
    response.content = b"{}"
    response.text = "nope"
    response.reason = "Bad Request"
    response.json.return_value = payload
    return response


def test_iter_issues_follows_next_page_token_until_last_page() -> None:
    session = MagicMock()
    session.request.side_effect = [
        _response(200, {"issues": [{"key": "PAY-1"}], "isLast": False, "nextPageToken": "tok-2"}),
        _response(200, {"issues": [{"key": "PAY-2"}], "isLast": True, "nextPageToken": "ignored"}),
    ]
    client = JiraCloudClient(
        _settings(),
        session=session,
        page_size=50,
        api_root="https://example.atlassian.net",
    )

    issues = list(client.iter_issues("project = PAY AND issuetype = Bug", ["summary", "description"]))

    assert [issue["key"] for issue in issues] == ["PAY-1", "PAY-2"]
    assert session.request.call_count == 2
    first_body = session.request.call_args_list[0].kwargs["json"]
    second_body = session.request.call_args_list[1].kwargs["json"]
    assert first_body["maxResults"] == 50
    assert "nextPageToken" not in first_body
    assert second_body["nextPageToken"] == "tok-2"
    assert first_body["fields"] == ["summary", "description"]


def test_repeated_page_token_stops_the_loop() -> None:
    session = MagicMock()
    session.request.side_effect = [
        _response(200, {"issues": [{"key": "PAY-1"}], "isLast": False, "nextPageToken": "same"}),
        _response(200, {"issues": [{"key": "PAY-2"}], "isLast": False, "nextPageToken": "same"}),
    ]
    client = JiraCloudClient(_settings(), session=session, api_root="https://example.atlassian.net")

    issues = list(client.iter_issues("project = PAY", ["summary"]))

    assert [issue["key"] for issue in issues] == ["PAY-1", "PAY-2"]
    assert session.request.call_count == 2


def test_create_issue_posts_project_fields() -> None:
    session = MagicMock()
    session.request.return_value = _response(201, {"key": "FDE-63"})
    client = JiraCloudClient(_settings(), session=session, api_root="https://example.atlassian.net")
    description = {"type": "doc", "version": 1, "content": []}

    key = client.create_issue(
        "FDE",
        "Flash sale oversell",
        description,
        "Bug",
        priority="High",
        labels=["triage-positive"],
    )

    assert key == "FDE-63"
    body = session.request.call_args.kwargs["json"]
    assert body["fields"]["project"] == {"key": "FDE"}
    assert body["fields"]["summary"] == "Flash sale oversell"
    assert body["fields"]["issuetype"] == {"name": "Bug"}
    assert body["fields"]["priority"] == {"name": "High"}
    assert body["fields"]["labels"] == ["triage-positive"]
    assert body["fields"]["description"] is description


def test_add_comment_posts_adf_paragraphs() -> None:
    session = MagicMock()
    session.request.return_value = _response(201, {"id": "10001"})
    client = JiraCloudClient(_settings(), session=session, api_root="https://example.atlassian.net")

    comment_id = client.add_comment("fde-64", "Triage recommendation\n\nDo not reuse a root cause.")

    assert comment_id == "10001"
    body = session.request.call_args.kwargs["json"]["body"]
    assert body["type"] == "doc"
    assert body["content"][0]["content"][0]["text"] == "Triage recommendation"
    assert body["content"][1]["content"][0]["text"] == "Do not reuse a root cause."
    session.request.assert_called_once_with(
        "POST",
        "https://example.atlassian.net/rest/api/3/issue/FDE-64/comment",
        json={"body": body},
        params=None,
        timeout=60,
    )


def test_get_issue_requests_one_key() -> None:
    session = MagicMock()
    session.request.return_value = _response(200, {"key": "FDE-3", "fields": {"summary": "Oversell"}})
    client = JiraCloudClient(_settings(), session=session, api_root="https://example.atlassian.net")

    issue = client.get_issue("fde-3", ["summary", "description"])

    assert issue["key"] == "FDE-3"
    session.request.assert_called_once_with(
        "GET",
        "https://example.atlassian.net/rest/api/3/issue/FDE-3",
        json=None,
        params={"fields": "summary,description"},
        timeout=60,
    )


def test_get_issue_rejects_a_malformed_key() -> None:
    session = MagicMock()
    client = JiraCloudClient(_settings(), session=session, api_root="https://example.atlassian.net")

    with pytest.raises(Exception, match="FDE-3"):
        client.get_issue("not a key", ["summary"])

    session.request.assert_not_called()


def test_api_error_includes_jira_message() -> None:
    session = MagicMock()
    session.request.return_value = _response(401, {"errorMessages": ["Client must be authenticated"]})
    client = JiraCloudClient(_settings(), session=session, api_root="https://example.atlassian.net")

    with pytest.raises(JiraApiError, match="Client must be authenticated"):
        client.list_fields()


def test_classic_token_uses_the_site_url() -> None:
    session = MagicMock()
    session.get.return_value = _response(200, {"displayName": "Analyst"})

    root = JiraApiRootResolver().resolve("https://example.atlassian.net", session)

    assert root == "https://example.atlassian.net"
    session.get.assert_called_once_with(
        "https://example.atlassian.net/rest/api/3/myself",
        timeout=60,
    )


def test_scoped_token_uses_the_atlassian_gateway() -> None:
    session = MagicMock()
    session.get.side_effect = [
        _response(401, {"message": "Client must be authenticated"}),
        _response(200, {"cloudId": "cloud-123"}),
        _response(200, {"displayName": "Analyst"}),
    ]

    root = JiraApiRootResolver().resolve("https://example.atlassian.net", session)

    assert root == "https://api.atlassian.com/ex/jira/cloud-123"


def test_scoped_token_without_jira_scopes_explains_the_fix() -> None:
    session = MagicMock()
    session.get.side_effect = [
        _response(401, {"message": "Client must be authenticated"}),
        _response(200, {"cloudId": "cloud-123"}),
        _response(401, {"message": "Unauthorized; scope does not match"}),
    ]

    with pytest.raises(JiraApiError, match="read:jira-work"):
        JiraApiRootResolver().resolve("https://example.atlassian.net", session)
