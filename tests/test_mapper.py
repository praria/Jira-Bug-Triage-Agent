"""Tests for mapping a Jira issue onto a Defect."""

from jira_triage.jira.mapper import DefectMapper
from jira_triage.jira.root_cause import DescriptionSectionParser


def _mapper() -> DefectMapper:
    return DefectMapper(DescriptionSectionParser())


def test_maps_issue_to_the_dump_shape() -> None:
    issue = {
        "key": "FDE-3",
        "fields": {
            "summary": "Inventory oversell during flash sale",
            "project": {"key": "FDE"},
            "priority": {"name": "Critical"},
            "description": {
                "type": "doc",
                "content": [
                    {
                        "type": "paragraph",
                        "content": [
                            {
                                "type": "text",
                                "text": "Multiple customers successfully purchased the same SKU despite stock reaching zero, resulting in unfulfillable orders.",
                            }
                        ],
                    },
                    {"type": "heading", "content": [{"type": "text", "text": "Logs"}]},
                    {
                        "type": "paragraph",
                        "content": [
                            {
                                "type": "text",
                                "text": "ERROR [InventoryService] - OptimisticLockException on SKU_9932: version mismatch",
                            }
                        ],
                    },
                    {"type": "heading", "content": [{"type": "text", "text": "Root Cause"}]},
                    {
                        "type": "paragraph",
                        "content": [
                            {
                                "type": "text",
                                "text": "Database transaction isolation level set to READ COMMITTED instead of SERIALIZABLE during high-concurrency checkout flow.",
                            }
                        ],
                    },
                ],
            },
        },
    }

    defect = _mapper().map_issue(issue)

    assert defect.to_dict() == {
        "summary": "Inventory oversell during flash sale",
        "description": "Multiple customers successfully purchased the same SKU despite stock reaching zero, resulting in unfulfillable orders.",
        "project_key": "FDE",
        "priority": "Critical",
        "labels": [
            "logs: ERROR [InventoryService] - OptimisticLockException on SKU_9932: version mismatch",
            "root_cause: Database transaction isolation level set to READ COMMITTED instead of SERIALIZABLE during high-concurrency checkout flow.",
        ],
    }


def test_blank_sections_leave_labels_empty() -> None:
    defect = _mapper().map_issue(
        {
            "key": "PAY-1",
            "fields": {"summary": "Blank", "description": None, "priority": None, "project": None},
        }
    )

    assert defect.logs == ""
    assert defect.root_cause == ""
    assert defect.description == ""
    assert defect.project_key == "PAY"
    assert defect.priority == ""
    assert defect.labels == []
