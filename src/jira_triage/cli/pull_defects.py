"""Pull defects and save them as JSON."""

from __future__ import annotations

import sys
from pathlib import Path

from jira_triage.bootstrap import build_defect_source
from jira_triage.config.settings import Settings
from jira_triage.dump import DefectDumpWriter
from jira_triage.errors import TriageError

DEFAULT_DUMP_NAME = "defectdump.json"


def run(env_file: Path | None = None, output_file: Path | None = None) -> int:
    """Pull defects and write them to ``output_file``. A one-line summary goes to stderr."""
    destination = output_file or Path(DEFAULT_DUMP_NAME)
    try:
        settings = Settings.from_env(env_file if env_file is not None and env_file.is_file() else None)
        defects = build_defect_source(settings).pull()
        DefectDumpWriter(destination).write(defects)
    except TriageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Pulled {len(defects)} defects into {destination}.", file=sys.stderr)
    return 0


def main() -> int:
    return run()
