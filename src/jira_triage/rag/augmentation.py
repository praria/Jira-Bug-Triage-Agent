"""Augmentation step: turn retrieved bugs into a recommendation and its prompt."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from typing import Protocol

from jira_triage.errors import TriageError
from jira_triage.rag.retrieval import SimilarDefect, RetrievalResult

# A stored bug at or above this cosine similarity is close enough to reuse.
CLEAR_MATCH_SIMILARITY = 0.60


@dataclass(frozen=True)
class TriageRecommendation:
    """The text shown on the dashboard and posted to Jira after approval."""

    defect_id: str
    clear_match: bool
    headline: str
    reasoning: str
    instructions: tuple[str, ...]
    notice: str = ""

    def comment_text(self) -> str:
        lines = [
            "Triage recommendation",
            "",
            self.headline,
            "",
            "Reasoning",
            self.reasoning,
            "",
            "What to do",
        ]
        lines.extend(f"{index}. {step}" for index, step in enumerate(self.instructions, start=1))
        return "\n".join(lines).strip() + "\n"


class RecommendationModel(Protocol):
    """Phrases a clear historical match. Weak matches do not use this."""

    def write(self, result: RetrievalResult) -> TriageRecommendation:
        """Return the recommendation for a clear match."""


class TriageAdvisor:
    """Choose a clear-match recommendation or investigation steps."""

    def __init__(
        self,
        model: RecommendationModel | None = None,
        threshold: float = CLEAR_MATCH_SIMILARITY,
    ) -> None:
        self._model = model
        self._threshold = threshold

    def advise(self, result: RetrievalResult) -> TriageRecommendation:
        if not _is_clear(result, self._threshold):
            return recommend_no_clear_match(result, self._threshold)
        if self._model is None:
            return recommend_clear_match(result, self._threshold)
        try:
            drafted = self._model.write(result)
        except TriageError as error:
            fallback = recommend_clear_match(result, self._threshold)
            return replace(fallback, notice=str(error))
        return replace(drafted, defect_id=result.defect.defect_id, clear_match=True, notice="")


class GeminiRecommendationModel:
    """Ask Gemini to phrase a clear match from the stored root cause."""

    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model

    def write(self, result: RetrievalResult) -> TriageRecommendation:
        raw = self._generate(clear_match_prompt(result))
        headline, reasoning, instructions = parse_model_recommendation(raw)
        return TriageRecommendation(
            defect_id=result.defect.defect_id,
            clear_match=True,
            headline=headline,
            reasoning=reasoning,
            instructions=instructions,
        )

    def _generate(self, prompt: str) -> str:
        try:
            from google import genai
        except ImportError as error:
            raise TriageError("Install google-genai to use GEMINI_API_KEY.") from error
        client = genai.Client(api_key=self._api_key)
        response = client.models.generate_content(model=self._model, contents=prompt)
        text = getattr(response, "text", None)
        if not text or not str(text).strip():
            raise TriageError("Gemini returned an empty recommendation.")
        return str(text)


def recommend_clear_match(result: RetrievalResult, threshold: float = CLEAR_MATCH_SIMILARITY) -> TriageRecommendation:
    """Recommend the closest stored root cause."""
    top = result.matches[0]
    others = _score_list(result.matches[1:])
    cause = top.root_cause.strip()
    title = _first_line(top.document) or top.defect_id
    percent = _percent(top.similarity)
    line = _percent(threshold)
    if cause:
        headline = (
            f"Treat this like {top.defect_id} ({percent} similar). "
            f"Confirm that root cause, then apply it here: {cause}"
        )
        confirm = f"Confirm this bug fails for the same reason: {cause}"
    else:
        headline = (
            f"{top.defect_id} is a clear match ({percent}), but it has no stored root cause to reuse."
        )
        confirm = f"Open {top.defect_id} and compare the failure with this ticket before choosing a fix."
    reasoning = (
        f"{result.defect.defect_id} matches {top.defect_id} at {percent}, "
        f"which is at or above the {line} line for a clear historical match. "
        f"The stored summary is: {title}. "
        f"The other nearest bugs are {others}. "
        f"Follow {top.defect_id}. Use the lower-ranked tickets only as context."
    )
    instructions = (
        f"Open {top.defect_id} and compare its logs with this ticket.",
        confirm,
        f"Apply that fix to {result.defect.defect_id}, then rerun the failing case.",
        "Write the confirmed root cause into the Root Cause section of this ticket.",
        "Leave the lower-ranked tickets unused when their symptoms differ.",
    )
    return TriageRecommendation(
        defect_id=result.defect.defect_id,
        clear_match=True,
        headline=headline,
        reasoning=reasoning,
        instructions=instructions,
    )


def recommend_no_clear_match(
    result: RetrievalResult,
    threshold: float = CLEAR_MATCH_SIMILARITY,
) -> TriageRecommendation:
    """Tell the reader not to reuse a historical root cause."""
    defect_id = result.defect.defect_id
    line = _percent(threshold)
    if not result.matches:
        return TriageRecommendation(
            defect_id=defect_id,
            clear_match=False,
            headline="No stored bugs are available to compare with this ticket.",
            reasoning=(
                "The local defect index has no other bugs, so there is no historical root cause to reuse."
            ),
            instructions=(
                "Investigate this bug from the description and logs already on the ticket.",
                "Write a Root Cause section only after that investigation.",
                "Index the ticket after the root cause is saved so a later bug can match it.",
            ),
        )
    top = result.matches[0]
    listed = _score_list(result.matches)
    nearest_ids = ", ".join(match.defect_id for match in result.matches)
    return TriageRecommendation(
        defect_id=defect_id,
        clear_match=False,
        headline=(
            f"No stored bug is close enough to reuse a root cause. "
            f"The nearest is {top.defect_id} at {_percent(top.similarity)}."
        ),
        reasoning=(
            f"A clear match requires at least {line} similarity between this bug's summary and description "
            f"and a stored bug. The nearest tickets are {listed}. "
            f"Those scores show shared wording, not evidence that their root causes explain {defect_id}. "
            f"Do not copy a fix from them."
        ),
        instructions=(
            f"Do not apply the root cause from {nearest_ids}.",
            "Reproduce the bug and add the current logs to the Logs section.",
            "Find the change, configuration, or data that produces this failure.",
            "Write a new Root Cause section on this ticket after that check.",
            "Index the ticket after the root cause is saved so a later similar bug can match it.",
        ),
    )


def clear_match_prompt(result: RetrievalResult) -> str:
    """Prompt for phrasing a clear match without inventing a root cause."""
    top = result.matches[0]
    allowed = ", ".join(match.defect_id for match in result.matches)
    blocks = [
        "You write a Jira defect triage recommendation.",
        "Return only JSON with keys headline, reasoning, and instructions.",
        "instructions is an array of 4 to 6 short steps.",
        f"The top match is a clear historical match. Recommend confirming and applying the root cause of {top.defect_id}.",
        "Do not invent a root cause that is not written below.",
        f"Mention only these defect ids: {allowed}.",
        "",
        f"New bug: {result.defect.defect_id}",
        f"Summary: {result.defect.summary}",
        f"Description: {result.defect.description}",
        f"Root cause already on the ticket: {result.defect.root_cause or 'none'}",
        "",
        "Nearest stored bugs:",
    ]
    for match in result.matches:
        blocks.append(
            f"- {match.defect_id} similarity {_percent(match.similarity)}; "
            f"summary {_first_line(match.document)}; root cause {match.root_cause or 'none'}"
        )
    return "\n".join(blocks)


def parse_model_recommendation(raw: str) -> tuple[str, str, tuple[str, ...]]:
    """Read the JSON object returned by the recommendation model."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise TriageError("Gemini returned a recommendation that was not valid JSON.") from error
    if not isinstance(payload, dict):
        raise TriageError("Gemini returned a recommendation that was not a JSON object.")
    headline = str(payload.get("headline") or "").strip()
    reasoning = str(payload.get("reasoning") or "").strip()
    steps = payload.get("instructions")
    if not headline or not reasoning or not isinstance(steps, list):
        raise TriageError("Gemini left the headline, reasoning, or instructions empty.")
    instructions = tuple(str(step).strip() for step in steps if str(step).strip())
    if len(instructions) < 2:
        raise TriageError("Gemini did not return enough investigation steps.")
    return headline, reasoning, instructions


def _is_clear(result: RetrievalResult, threshold: float) -> bool:
    return bool(result.matches) and result.matches[0].similarity >= threshold


def _percent(similarity: float) -> str:
    return f"{similarity * 100:.1f}%"


def _first_line(document: str) -> str:
    for line in document.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _score_list(matches: list[SimilarDefect]) -> str:
    if not matches:
        return "none"
    return ", ".join(f"{match.defect_id} ({_percent(match.similarity)})" for match in matches)
