"""Embed pulled defects and store them in ChromaDB."""

from __future__ import annotations

import sys
from pathlib import Path

from jira_triage.bootstrap import build_defect_indexer
from jira_triage.config.settings import Settings
from jira_triage.errors import TriageError


def run(env_file: Path | None = None, chroma_path: Path | None = None) -> int:
    """Pull defects, embed summary and description, and upsert the vectors."""
    try:
        settings = Settings.from_env(env_file if env_file is not None and env_file.is_file() else None)
        destination = chroma_path or Path(settings.chroma_path)
        count = build_defect_indexer(settings, destination).index()
    except TriageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Indexed {count} defects into {destination}.", file=sys.stderr)
    return 0


def main() -> int:
    return run()
