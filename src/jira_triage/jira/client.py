"""HTTP client for the Jira Cloud REST API."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import requests

from jira_triage.config.settings import Settings
from jira_triage.errors import JiraApiError, TriageError
from jira_triage.jira.adf import plain_text_to_adf
from jira_triage.jira.issue_key import require_defect_id


_SCOPE_HELP = (
    "This API token is scoped, and those scopes do not include Jira access. "
    "Create an API token with scopes read:jira-work, read:jira-user, and write:jira-work, "
    "replace JIRA_API_TOKEN in .env, and run python pull_defects.py again."
)


class JiraApiRootResolver:
    """Pick the REST base URL for the token in use.

    A classic API token calls ``https://<site>.atlassian.net``. A scoped API token
    calls ``https://api.atlassian.com/ex/jira/<cloudId>``.
    """

    def __init__(self, timeout: float = 60) -> None:
        self._timeout = timeout

    def resolve(self, site_url: str, session: requests.Session) -> str:
        site_url = site_url.rstrip("/")
        if _is_authenticated(session, f"{site_url}/rest/api/3/myself", self._timeout):
            return site_url

        cloud_id = self._cloud_id(session, site_url)
        gateway = f"https://api.atlassian.com/ex/jira/{cloud_id}"
        response = session.get(f"{gateway}/rest/api/3/myself", timeout=self._timeout)
        if response.status_code == 200:
            return gateway

        message = _error_message(response)
        if response.status_code == 401 and "scope does not match" in message.casefold():
            raise JiraApiError(401, _SCOPE_HELP, "/rest/api/3/myself")
        detail = message or "Jira rejected the email or API token."
        raise JiraApiError(response.status_code, detail, "/rest/api/3/myself")

    def _cloud_id(self, session: requests.Session, site_url: str) -> str:
        response = session.get(f"{site_url}/_edge/tenant_info", timeout=self._timeout)
        if response.status_code >= 400:
            raise JiraApiError(response.status_code, _error_message(response), "/_edge/tenant_info")
        body = response.json() if response.content else {}
        cloud_id = body.get("cloudId") if isinstance(body, dict) else None
        if not cloud_id:
            raise JiraApiError(
                response.status_code,
                "Jira did not return a cloud id for this site.",
                "/_edge/tenant_info",
            )
        return str(cloud_id)


def _is_authenticated(session: requests.Session, url: str, timeout: float) -> bool:
    response = session.get(url, timeout=timeout)
    return response.status_code == 200


class JiraCloudClient:
    """Call Jira Cloud with basic auth (email and API token).

    Issue search uses ``POST /rest/api/3/search/jql`` and follows ``nextPageToken``.
    """

    def __init__(
        self,
        settings: Settings,
        session: requests.Session | None = None,
        page_size: int = 100,
        timeout: float = 60,
        api_root: str | None = None,
        root_resolver: JiraApiRootResolver | None = None,
    ) -> None:
        if page_size < 1:
            raise ValueError("page_size must be at least 1")
        self._site_url = settings.jira_base_url
        self._api_root = api_root.rstrip("/") if api_root else None
        self._root_resolver = root_resolver or JiraApiRootResolver(timeout)
        self._page_size = page_size
        self._timeout = timeout
        self._session = session or requests.Session()
        self._session.auth = (settings.jira_email, settings.jira_api_token)
        self._session.headers.update(
            {
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )

    def create_issue(
        self,
        project_key: str,
        summary: str,
        description: dict[str, Any],
        issue_type: str,
        priority: str | None = None,
        labels: list[str] | None = None,
    ) -> str:
        """Create one issue and return its key."""
        fields: dict[str, Any] = {
            "project": {"key": project_key},
            "summary": summary,
            "description": description,
            "issuetype": {"name": issue_type},
        }
        if priority:
            fields["priority"] = {"name": priority}
        if labels:
            fields["labels"] = labels
        payload = self._request("POST", "/rest/api/3/issue", json_body={"fields": fields})
        key = payload.get("key") if isinstance(payload, dict) else None
        if not key:
            raise JiraApiError(201, "Jira did not return an issue key.", "/rest/api/3/issue")
        return str(key)

    def add_comment(self, issue_key: str, body: str) -> str:
        """Post a plain-text comment on one issue and return the comment id."""
        key = require_defect_id(issue_key)
        text = body.strip()
        if not text:
            raise TriageError("The recommendation comment is empty.")
        payload = self._request(
            "POST",
            f"/rest/api/3/issue/{key}/comment",
            json_body={"body": plain_text_to_adf(text)},
        )
        comment_id = payload.get("id") if isinstance(payload, dict) else None
        if not comment_id:
            raise JiraApiError(201, "Jira did not return a comment id.", f"/rest/api/3/issue/{key}/comment")
        return str(comment_id)

    def get_issue(self, issue_key: str, fields: list[str]) -> dict[str, Any]:
        """Return one issue by key, requesting only ``fields``."""
        key = require_defect_id(issue_key)
        payload = self._request(
            "GET",
            f"/rest/api/3/issue/{key}",
            params={"fields": ",".join(fields)},
        )
        if not isinstance(payload, dict):
            raise JiraApiError(200, "Jira issue response was not a JSON object.", f"/rest/api/3/issue/{key}")
        return payload

    def list_fields(self) -> list[dict[str, Any]]:
        """Return every field visible to the authenticated user."""
        payload = self._request("GET", "/rest/api/3/field")
        if not isinstance(payload, list):
            raise JiraApiError(200, "Jira field list was not a JSON array.", "/rest/api/3/field")
        return payload

    def iter_issues(self, jql: str, fields: list[str]) -> Iterator[dict[str, Any]]:
        """Yield every issue matching ``jql``, requesting only ``fields``."""
        token: str | None = None
        seen_tokens: set[str] = set()
        while True:
            body: dict[str, Any] = {
                "jql": jql,
                "maxResults": self._page_size,
                "fields": fields,
            }
            if token is not None:
                if token in seen_tokens:
                    return
                seen_tokens.add(token)
                body["nextPageToken"] = token

            page = self._request("POST", "/rest/api/3/search/jql", json_body=body)
            if not isinstance(page, dict):
                raise JiraApiError(
                    200,
                    "Jira search response was not a JSON object.",
                    "/rest/api/3/search/jql",
                )
            issues = page.get("issues") or []
            if not isinstance(issues, list):
                raise JiraApiError(
                    200,
                    "Jira search response did not include an issues array.",
                    "/rest/api/3/search/jql",
                )
            yield from issues

            if page.get("isLast") is True:
                return
            next_token = page.get("nextPageToken")
            if not next_token:
                return
            token = str(next_token)

    def _request(
        self,
        method: str,
        path: str,
        json_body: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
    ) -> Any:
        if self._api_root is None:
            self._api_root = self._root_resolver.resolve(self._site_url, self._session)
        response = self._session.request(
            method,
            f"{self._api_root}{path}",
            json=json_body,
            params=params,
            timeout=self._timeout,
        )
        if response.status_code >= 400:
            raise JiraApiError(response.status_code, _error_message(response), path)
        if response.status_code == 204 or not response.content:
            return {}
        return response.json()


def _error_message(response: requests.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return (response.text or response.reason or "Unknown Jira error")[:500]
    if isinstance(body, dict):
        parts: list[str] = []
        if body.get("message"):
            parts.append(str(body["message"]))
        parts.extend(str(message) for message in body.get("errorMessages") or [])
        errors = body.get("errors") or {}
        if isinstance(errors, dict):
            parts.extend(f"{key}: {value}" for key, value in errors.items())
        if parts:
            return "; ".join(parts)
    text = response.text or response.reason or "Unknown Jira error"
    return text[:500]
