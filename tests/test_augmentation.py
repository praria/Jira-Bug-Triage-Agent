"""Tests for clear-match recommendations and weak-match instructions."""

import pytest

from jira_triage.domain.defect import Defect
from jira_triage.errors import TriageError
from jira_triage.rag.augmentation import (
    CLEAR_MATCH_SIMILARITY,
    GeminiRecommendationModel,
    TriageAdvisor,
    TriageRecommendation,
    parse_model_recommendation,
)
from jira_triage.rag.retrieval import RetrievalResult, SimilarDefect


def _defect(defect_id: str = "FDE-63") -> Defect:
    return Defect(
        defect_id=defect_id,
        summary="Flash sale checkout sells the last unit twice",
        description="Two customers bought the same SKU after stock reached zero.",
        project_key="FDE",
        priority="High",
        logs="OptimisticLockException",
        root_cause="",
    )


def _match(defect_id: str, similarity: float, root_cause: str) -> SimilarDefect:
    return SimilarDefect(
        defect_id=defect_id,
        root_cause=root_cause,
        document=f"{defect_id} summary\n\nDetails",
        similarity=similarity,
    )


def _result(matches: list[SimilarDefect], defect_id: str = "FDE-63") -> RetrievalResult:
    return RetrievalResult(defect=_defect(defect_id), matches=matches)


def test_clear_match_recommends_the_top_root_cause() -> None:
    result = _result(
        [
            _match("FDE-3", 0.765, "Checkout uses READ COMMITTED."),
            _match("FDE-54", 0.493, "Buy Again skips availability."),
            _match("FDE-28", 0.450, "Bulk quantity overflows the stack."),
        ]
    )

    recommendation = TriageAdvisor().advise(result)

    assert recommendation.clear_match is True
    assert "FDE-3" in recommendation.headline
    assert "READ COMMITTED" in recommendation.headline
    assert "76.5%" in recommendation.reasoning
    assert "FDE-54 (49.3%)" in recommendation.reasoning
    assert any("READ COMMITTED" in step for step in recommendation.instructions)
    comment = recommendation.comment_text()
    assert comment.startswith("Triage recommendation")
    assert "Reasoning" in comment
    assert "1. " in comment


def test_weak_match_refuses_the_historical_root_cause() -> None:
    result = _result(
        [
            _match("FDE-35", 0.149, "The return service timed out."),
            _match("FDE-55", 0.145, "The session timer cleared the cart."),
            _match("FDE-18", 0.132, "Guest email collided."),
        ],
        defect_id="FDE-64",
    )

    recommendation = TriageAdvisor().advise(result)

    assert recommendation.clear_match is False
    assert "No stored bug is close enough" in recommendation.headline
    assert "FDE-35" in recommendation.headline
    assert "14.9%" in recommendation.headline
    assert "Do not copy a fix" in recommendation.reasoning
    assert recommendation.instructions[0].startswith("Do not apply the root cause")
    assert "FDE-35" in recommendation.instructions[0]
    assert "return service" not in recommendation.headline
    assert result.matches[0].similarity < CLEAR_MATCH_SIMILARITY


def test_empty_history_asks_for_a_fresh_investigation() -> None:
    recommendation = TriageAdvisor().advise(_result([]))

    assert recommendation.clear_match is False
    assert "No stored bugs" in recommendation.headline
    assert any("Index the ticket" in step for step in recommendation.instructions)


class FakeModel:
    def __init__(self, recommendation: TriageRecommendation | None = None, error: Exception | None = None) -> None:
        self._recommendation = recommendation
        self._error = error
        self.calls = 0

    def write(self, result: RetrievalResult) -> TriageRecommendation:
        self.calls += 1
        if self._error:
            raise self._error
        assert self._recommendation is not None
        return self._recommendation


def test_gemini_phrases_a_clear_match_and_is_skipped_for_a_weak_match() -> None:
    drafted = TriageRecommendation(
        defect_id="IGNORED",
        clear_match=False,
        headline="Apply the READ COMMITTED fix from FDE-3.",
        reasoning="FDE-3 describes the same flash-sale oversell.",
        instructions=("Compare logs with FDE-3.", "Change isolation to SERIALIZABLE."),
    )
    model = FakeModel(drafted)
    advisor = TriageAdvisor(model)
    clear = _result([_match("FDE-3", 0.765, "Checkout uses READ COMMITTED.")])
    weak = _result([_match("FDE-35", 0.149, "Timeout.")], defect_id="FDE-64")

    phrased = advisor.advise(clear)
    refused = advisor.advise(weak)

    assert model.calls == 1
    assert phrased.defect_id == "FDE-63"
    assert phrased.clear_match is True
    assert phrased.headline == drafted.headline
    assert refused.clear_match is False
    assert "No stored bug is close enough" in refused.headline


def test_model_failure_falls_back_to_the_stored_root_cause() -> None:
    model = FakeModel(error=TriageError("Gemini returned an empty recommendation."))
    result = _result([_match("FDE-3", 0.80, "Checkout uses READ COMMITTED.")])

    recommendation = TriageAdvisor(model).advise(result)

    assert recommendation.clear_match is True
    assert "READ COMMITTED" in recommendation.headline
    assert recommendation.notice == "Gemini returned an empty recommendation."


def test_parse_model_recommendation_reads_json_and_fences() -> None:
    headline, reasoning, instructions = parse_model_recommendation(
        """```json
        {"headline": "Apply FDE-3.", "reasoning": "Same oversell.", "instructions": ["Compare logs.", "Retest."]}
        ```"""
    )

    assert headline == "Apply FDE-3."
    assert reasoning == "Same oversell."
    assert instructions == ("Compare logs.", "Retest.")


def test_parse_model_recommendation_rejects_incomplete_json() -> None:
    with pytest.raises(TriageError, match="not valid JSON"):
        parse_model_recommendation("not json")


def test_gemini_model_uses_the_injected_response(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _result([_match("FDE-3", 0.80, "Checkout uses READ COMMITTED.")])

    def generate(self, prompt: str) -> str:
        assert "FDE-3" in prompt
        assert "READ COMMITTED" in prompt
        return '{"headline": "Apply FDE-3.", "reasoning": "Same bug.", "instructions": ["Compare.", "Retest."]}'

    monkeypatch.setattr(GeminiRecommendationModel, "_generate", generate)
    recommendation = GeminiRecommendationModel("test-key", "gemini-3.5-flash").write(result)

    assert recommendation.headline == "Apply FDE-3."
    assert recommendation.instructions == ("Compare.", "Retest.")
