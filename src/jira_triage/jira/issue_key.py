"""Validate a Jira issue key before it is used in a URL or a lookup."""

from __future__ import annotations

import re

from jira_triage.errors import TriageError

_DEFECT_ID = re.compile(r"^[A-Z][A-Z0-9]+-\d+$")


def require_defect_id(value: str) -> str:
    """Return a normalized Jira issue key such as ``FDE-3``."""
    key = value.strip().upper()
    if _DEFECT_ID.fullmatch(key) is None:
        raise TriageError("Enter a Jira defect id such as FDE-3.")
    return key
