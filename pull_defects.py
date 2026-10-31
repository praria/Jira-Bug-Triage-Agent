"""Pull Jira bugs and save them to defectdump.json.

Run from the project root after filling in .env:

    python pull_defects.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from jira_triage.cli.pull_defects import run  # noqa: E402


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    raise SystemExit(run(root / ".env", root / "defectdump.json"))
