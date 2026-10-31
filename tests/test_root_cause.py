"""Tests for splitting Logs and Root Cause out of a description."""

from jira_triage.jira.root_cause import DescriptionSectionParser


def test_splits_logs_and_root_cause_from_the_narrative() -> None:
    description = "\n".join(
        [
            "Checkout fails for USD.",
            "",
            "Logs:",
            "ERROR [TaxClient] - Read timed out",
            "",
            "Root Cause:",
            "Currency code is null.",
            "",
            "Expected:",
            "Charge succeeds.",
        ]
    )

    parts = DescriptionSectionParser().split(description)

    assert parts.description == "Checkout fails for USD.\n\nExpected:\nCharge succeeds."
    assert parts.logs == "ERROR [TaxClient] - Read timed out"
    assert parts.root_cause == "Currency code is null."


def test_reads_headings_and_stops_at_the_next_heading() -> None:
    document = {
        "type": "doc",
        "content": [
            {"type": "paragraph", "content": [{"type": "text", "text": "Invoice tax is wrong."}]},
            {"type": "heading", "content": [{"type": "text", "text": "Logs"}]},
            {
                "type": "paragraph",
                "content": [{"type": "text", "text": "WARN [LoyaltyConsumer] - deserialization failed"}],
            },
            {"type": "heading", "content": [{"type": "text", "text": "Root Cause"}]},
            {"type": "paragraph", "content": [{"type": "text", "text": "Tax rounds down."}]},
            {"type": "heading", "content": [{"type": "text", "text": "Expected"}]},
            {"type": "paragraph", "content": [{"type": "text", "text": "Tax rounds half up."}]},
        ],
    }

    parts = DescriptionSectionParser("Root cause").split(document)

    assert parts.logs == "WARN [LoyaltyConsumer] - deserialization failed"
    assert parts.root_cause == "Tax rounds down."
    assert parts.description == "Invoice tax is wrong.\n\nExpected\n\nTax rounds half up."


def test_inline_label_is_case_insensitive() -> None:
    parts = DescriptionSectionParser().split("Payment hangs.\n\nroot cause: gateway timeout")

    assert parts.root_cause == "gateway timeout"
    assert parts.logs == ""
    assert parts.description == "Payment hangs."


def test_missing_sections_stay_empty() -> None:
    parts = DescriptionSectionParser().split("Payment hangs.")

    assert parts.description == "Payment hangs."
    assert parts.logs == ""
    assert parts.root_cause == ""
