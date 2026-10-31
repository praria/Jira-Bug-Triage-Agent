"""Split Logs and Root Cause sections out of a Jira description."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from jira_triage.jira.adf import AdfPlainTextConverter


@dataclass(frozen=True)
class DescriptionParts:
    """Narrative text plus the Logs and Root Cause sections."""

    description: str
    logs: str
    root_cause: str


class DescriptionSectionParser:
    """Pull the Logs and Root Cause sections out of a description.

    Each label may be a heading or a line such as ``Root Cause:``. The
    surrounding text stays in ``description``. A missing section is an empty
    string.
    """

    def __init__(
        self,
        root_cause_label: str = "Root Cause",
        logs_label: str = "Logs",
        text_converter: AdfPlainTextConverter | None = None,
    ) -> None:
        self._sections = (
            ("logs", _header_pattern(logs_label)),
            ("root_cause", _header_pattern(root_cause_label)),
        )
        self._text = text_converter or AdfPlainTextConverter()

    def split(self, description: Any) -> DescriptionParts:
        if isinstance(description, dict) and description.get("type") == "doc":
            parts = self._split_document(description)
            if parts.logs or parts.root_cause:
                return parts
            return self._split_text(self._text.convert(description).strip())
        return self._split_text(self._text.convert(description).strip())

    def _split_text(self, text: str) -> DescriptionParts:
        if not text:
            return DescriptionParts("", "", "")
        body: list[str] = []
        captured: dict[str, list[str]] = {"logs": [], "root_cause": []}
        current: str | None = None
        for line in text.splitlines():
            matched = self._match_section(line)
            if matched is not None:
                current, inline = matched
                if inline:
                    captured[current].append(inline)
                continue
            if current is not None and self._is_other_header(line):
                current = None
            if current is None:
                body.append(line)
            else:
                captured[current].append(line)
        return DescriptionParts(
            _clean_body(body),
            _one_line(captured["logs"]),
            _one_line(captured["root_cause"]),
        )

    def _split_document(self, document: dict[str, Any]) -> DescriptionParts:
        body: list[Any] = []
        captured: dict[str, list[Any]] = {"logs": [], "root_cause": []}
        current: str | None = None
        for block in document.get("content") or []:
            matched = self._match_block(block)
            if matched is not None:
                current = matched
                inline = self._inline_after_label(block)
                if inline:
                    captured[current].append(inline)
                continue
            if current is not None and self._is_other_header_block(block):
                current = None
            if current is None:
                body.append(block)
            else:
                captured[current].append(block)
        return DescriptionParts(
            self._blocks_to_text(body),
            _one_line_text(self._blocks_to_text(captured["logs"])),
            _one_line_text(self._blocks_to_text(captured["root_cause"])),
        )

    def _match_section(self, line: str) -> tuple[str, str] | None:
        for key, pattern in self._sections:
            match = pattern.match(line)
            if match is not None:
                return key, match.group(1).strip()
        return None

    def _match_block(self, block: Any) -> str | None:
        if not isinstance(block, dict):
            return None
        matched = self._match_section(self._first_line(block))
        if matched is None:
            return None
        return matched[0]

    def _is_other_header_block(self, block: Any) -> bool:
        if not isinstance(block, dict) or self._match_block(block) is not None:
            return False
        if block.get("type") == "heading":
            return True
        return self._is_other_header(self._first_line(block))

    def _is_other_header(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped.endswith(":") or self._match_section(stripped) is not None:
            return False
        label = stripped[:-1].strip()
        if not label or len(label) > 80:
            return False
        return len(label.split()) <= 8

    def _inline_after_label(self, block: dict[str, Any]) -> str:
        lines = self._text.convert(block).strip().splitlines()
        if not lines:
            return ""
        matched = self._match_section(lines[0])
        if matched is None:
            return "\n".join(lines).strip()
        parts: list[str] = []
        if matched[1]:
            parts.append(matched[1])
        parts.extend(lines[1:])
        return "\n".join(parts).strip()

    def _first_line(self, block: dict[str, Any]) -> str:
        text = self._text.convert(block).strip()
        if not text:
            return ""
        return text.splitlines()[0]

    def _blocks_to_text(self, blocks: list[Any]) -> str:
        parts: list[str] = []
        for block in blocks:
            if isinstance(block, str):
                stripped = block.strip()
                if stripped:
                    parts.append(stripped)
                continue
            text = self._text.convert(block).strip()
            if text:
                parts.append(text)
        return "\n\n".join(parts).strip()


def _header_pattern(label: str) -> re.Pattern[str]:
    cleaned = " ".join(label.split())
    if not cleaned:
        raise ValueError("Section label is empty.")
    words = [re.escape(word) for word in cleaned.split()]
    return re.compile(r"^\s*" + r"\s+".join(words) + r"\s*:?\s*(.*)$", re.IGNORECASE)


def _clean_body(lines: list[str]) -> str:
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _one_line(lines: list[str]) -> str:
    return _one_line_text("\n".join(lines))


def _one_line_text(text: str) -> str:
    return " ".join(text.split())
