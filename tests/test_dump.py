"""Tests for writing the defect JSON dump."""

import json
from pathlib import Path

from jira_triage.domain.defect import Defect
from jira_triage.dump import DefectDumpWriter


def test_writes_a_json_array(tmp_path: Path) -> None:
    output = tmp_path / "defectdump.json"
    defects = [
        Defect(
            defect_id="FDE-3",
            summary="Inventory oversell",
            description="Stock went negative.",
            project_key="FDE",
            priority="Critical",
            logs="ERROR [InventoryService] - version mismatch",
            root_cause="Isolation level",
        ),
    ]

    DefectDumpWriter(output).write(defects)

    assert json.loads(output.read_text(encoding="utf-8")) == [
        {
            "summary": "Inventory oversell",
            "description": "Stock went negative.",
            "project_key": "FDE",
            "priority": "Critical",
            "labels": [
                "logs: ERROR [InventoryService] - version mismatch",
                "root_cause: Isolation level",
            ],
        }
    ]
