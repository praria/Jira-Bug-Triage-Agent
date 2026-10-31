"""Write pulled defects to a JSON file."""

from __future__ import annotations

import json
from pathlib import Path

from jira_triage.domain.defect import Defect


class DefectDumpWriter:
    """Save defects as a JSON array."""

    def __init__(self, output_file: Path) -> None:
        self._output_file = output_file

    def write(self, defects: list[Defect]) -> None:
        payload = [defect.to_dict() for defect in defects]
        text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        self._output_file.write_text(text, encoding="utf-8")
