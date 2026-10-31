"""Embed Jira defects and store them in the local Chroma database.

Run from the project root after filling in .env:

    python index_defects.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from jira_triage.cli.index_defects import run  # noqa: E402


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    raise SystemExit(run(root / ".env", root / ".chroma"))
