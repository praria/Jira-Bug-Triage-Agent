"""Tests for Atlassian Document Format conversion."""

from jira_triage.jira.adf import AdfPlainTextConverter


def test_converts_paragraphs_lists_and_breaks() -> None:
    document = {
        "type": "doc",
        "version": 1,
        "content": [
            {
                "type": "paragraph",
                "content": [
                    {"type": "text", "text": "Login fails"},
                    {"type": "hardBreak"},
                    {"type": "text", "text": "after deploy"},
                ],
            },
            {
                "type": "bulletList",
                "content": [
                    {
                        "type": "listItem",
                        "content": [
                            {"type": "paragraph", "content": [{"type": "text", "text": "Null token"}]}
                        ],
                    }
                ],
            },
            {
                "type": "orderedList",
                "content": [
                    {
                        "type": "listItem",
                        "content": [
                            {"type": "paragraph", "content": [{"type": "text", "text": "Retry"}]}
                        ],
                    }
                ],
            },
        ],
    }

    text = AdfPlainTextConverter().convert(document)

    assert text == "Login fails\nafter deploy\n- Null token\n1. Retry"


def test_empty_description_is_blank() -> None:
    converter = AdfPlainTextConverter()
    assert converter.convert(None) == ""
    assert converter.convert("") == ""


def test_plain_string_description_is_kept() -> None:
    assert AdfPlainTextConverter().convert("Already plain") == "Already plain"
