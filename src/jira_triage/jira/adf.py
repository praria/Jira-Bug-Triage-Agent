"""Convert Jira's Atlassian Document Format into plain text."""

from __future__ import annotations

import re
from typing import Any

_BLOCK_TYPES = frozenset({"paragraph", "heading", "codeBlock", "blockquote", "panel"})


class AdfPlainTextConverter:
    """Walk an ADF document and return readable plain text."""

    def convert(self, node: Any) -> str:
        text = self._walk(node)
        lines = [line.rstrip() for line in text.splitlines()]
        cleaned: list[str] = []
        previous_blank = False
        for line in lines:
            blank = line == ""
            if blank and previous_blank:
                continue
            cleaned.append(line)
            previous_blank = blank
        return "\n".join(cleaned).strip()

    def _walk(self, node: Any) -> str:
        if node is None:
            return ""
        if isinstance(node, str):
            return node
        if isinstance(node, list):
            return "".join(self._walk(child) for child in node)
        if not isinstance(node, dict):
            return ""

        node_type = node.get("type")
        if node_type == "text":
            return str(node.get("text") or "")
        if node_type == "hardBreak":
            return "\n"
        if node_type == "mention":
            return str((node.get("attrs") or {}).get("text") or "")
        if node_type == "emoji":
            attrs = node.get("attrs") or {}
            return str(attrs.get("text") or attrs.get("shortName") or "")
        if node_type == "inlineCard":
            return str((node.get("attrs") or {}).get("url") or "")
        if node_type == "bulletList":
            return self._render_list(node, ordered=False)
        if node_type == "orderedList":
            return self._render_list(node, ordered=True)

        inner = "".join(self._walk(child) for child in node.get("content") or [])
        if node_type in _BLOCK_TYPES:
            return inner.rstrip() + "\n"
        return inner

    def _render_list(self, node: dict[str, Any], ordered: bool) -> str:
        rendered: list[str] = []
        for index, item in enumerate(node.get("content") or [], start=1):
            body = self._walk(item).strip()
            if not body:
                continue
            prefix = f"{index}. " if ordered else "- "
            item_lines = body.splitlines()
            rendered.append(prefix + item_lines[0])
            pad = " " * len(prefix)
            rendered.extend(pad + line for line in item_lines[1:])
        if not rendered:
            return ""
        return "\n".join(rendered) + "\n"


def plain_text_to_adf(text: str) -> dict[str, Any]:
    """Turn a plain-text comment into an Atlassian Document Format body."""
    blocks: list[dict[str, Any]] = []
    for paragraph in re.split(r"\n\s*\n", text.strip()):
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if not lines:
            continue
        content: list[dict[str, Any]] = []
        for index, line in enumerate(lines):
            if index:
                content.append({"type": "hardBreak"})
            content.append({"type": "text", "text": line})
        blocks.append({"type": "paragraph", "content": content})
    if not blocks:
        blocks.append({"type": "paragraph", "content": [{"type": "text", "text": "Triage recommendation"}]})
    return {"type": "doc", "version": 1, "content": blocks}
