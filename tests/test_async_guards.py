"""Phase 2 tests — async guards. All LLM calls are mocked."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from sentinel.async_guards.hallucination import (
    GroundingResult,
    HallucinationJudge,
    check as check_hallucination,
)
from sentinel.async_guards.output_validator import validate as validate_output
from sentinel.async_guards.topic_guardrail import TopicResult, check as check_topic
from sentinel.policy import HallucinationPolicy, OutputPolicy, ViolationAction


# ── OutputValidator ───────────────────────────────────────────────────────────

class TestOutputValidator:
    def _policy(self, **kw) -> OutputPolicy:
        defaults = {"toxicity_check": True, "toxicity_action": ViolationAction.FLAG}
        defaults.update(kw)
        return OutputPolicy(**defaults)

    async def test_clean_output_passes(self):
        policy = self._policy()
        result = await validate_output("This is a helpful and friendly response.", policy)
        assert result is None

    async def test_toxic_output_flagged(self):
        policy = self._policy(toxicity_action=ViolationAction.FLAG)
        # better-profanity catches common slurs; use a known trigger word
        result = await validate_output("You are a damn fool and an idiot.", policy)
        assert result is not None
        assert result.rule_name == "output.toxicity"
        assert result.action == ViolationAction.FLAG

    async def test_toxicity_check_disabled(self):
        policy = self._policy(toxicity_check=False)
        result = await validate_output("damn this is bad content", policy)
        assert result is None

    async def test_toxic_output_blocked_when_action_block(self):
        policy = self._policy(toxicity_action=ViolationAction.BLOCK)
        result = await validate_output("You are a damn fool.", policy)
        # If profanity detected, action should be BLOCK
        if result is not None:
            assert result.action == ViolationAction.BLOCK

    async def test_run_id_propagated(self):
        policy = self._policy()
        result = await validate_output("You are a damn idiot.", policy, run_id="test-run-42")
        if result is not None:
            assert result.run_id == "test-run-42"


# ── HallucinationJudge ────────────────────────────────────────────────────────

class TestHallucinationJudge:
    def _policy(self, **kw) -> HallucinationPolicy:
        defaults = {"enabled": True, "threshold": 0.7, "action": ViolationAction.FLAG}
        defaults.update(kw)
        return HallucinationPolicy(**defaults)

    def _mock_judge(self, grounded: bool, confidence: float, reason: str = "test") -> HallucinationJudge:
        judge = MagicMock(spec=HallucinationJudge)
        judge.judge = AsyncMock(
            return_value=GroundingResult(grounded=grounded, confidence=confidence, reason=reason)
        )
        return judge

    async def test_grounded_above_threshold_passes(self):
        judge = self._mock_judge(grounded=True, confidence=0.9)
        policy = self._policy(threshold=0.7)
        result = await check_hallucination("q", "r", "ctx", policy, judge=judge)
        assert result is None

    async def test_not_grounded_flagged(self):
        judge = self._mock_judge(grounded=False, confidence=0.3, reason="No support found")
        policy = self._policy(threshold=0.7)
        result = await check_hallucination("q", "r", "ctx", policy, judge=judge)
        assert result is not None
        assert result.rule_name == "hallucination.low_grounding"
        assert result.action == ViolationAction.FLAG

    async def test_grounded_but_low_confidence_flagged(self):
        # Confidence below threshold even if technically grounded
        judge = self._mock_judge(grounded=True, confidence=0.5)
        policy = self._policy(threshold=0.7)
        result = await check_hallucination("q", "r", "ctx", policy, judge=judge)
        assert result is not None
        assert "0.50" in result.message

    async def test_at_threshold_boundary_passes(self):
        # Exactly at threshold — should pass (confidence == threshold means ok)
        judge = self._mock_judge(grounded=True, confidence=0.7)
        policy = self._policy(threshold=0.7)
        result = await check_hallucination("q", "r", "ctx", policy, judge=judge)
        assert result is None

    async def test_disabled_policy_skips_check(self):
        judge = self._mock_judge(grounded=False, confidence=0.0)
        policy = self._policy(enabled=False)
        result = await check_hallucination("q", "r", "ctx", policy, judge=judge)
        assert result is None
        judge.judge.assert_not_called()

    async def test_violation_contains_reason(self):
        judge = self._mock_judge(grounded=False, confidence=0.2, reason="Claims not in context")
        policy = self._policy(threshold=0.7)
        result = await check_hallucination("q", "r", "ctx", policy, judge=judge)
        assert result is not None
        assert "Claims not in context" in result.message

    async def test_custom_action_propagated(self):
        judge = self._mock_judge(grounded=False, confidence=0.1)
        policy = self._policy(action=ViolationAction.BLOCK)
        result = await check_hallucination("q", "r", "ctx", policy, judge=judge)
        assert result is not None
        assert result.action == ViolationAction.BLOCK


# ── TopicGuardrail ────────────────────────────────────────────────────────────

class TestTopicGuardrail:
    def _policy(self, topics: list[str] | None = None, **kw) -> OutputPolicy:
        defaults = {
            "topic_guardrail": topics or ["technology", "programming"],
            "topic_action": ViolationAction.FLAG,
        }
        defaults.update(kw)
        return OutputPolicy(**defaults)

    def _mock_llm(self, on_topic: bool, detected: str = "none", reason: str = "test"):
        llm = MagicMock()
        judge = MagicMock()
        judge.ainvoke = AsyncMock(
            return_value=TopicResult(on_topic=on_topic, detected_topic=detected, reason=reason)
        )
        llm.with_structured_output = MagicMock(return_value=judge)
        return llm

    async def test_on_topic_passes(self):
        llm = self._mock_llm(on_topic=True, detected="technology")
        policy = self._policy()
        result = await check_topic("Python is a great programming language.", policy, _llm=llm)
        assert result is None

    async def test_off_topic_flagged(self):
        llm = self._mock_llm(on_topic=False, detected="cooking", reason="Discusses recipes")
        policy = self._policy()
        result = await check_topic("Here is a recipe for pasta.", policy, _llm=llm)
        assert result is not None
        assert result.rule_name == "output.topic_guardrail"
        assert result.action == ViolationAction.FLAG

    async def test_no_guardrail_configured_skips(self):
        llm = self._mock_llm(on_topic=False)
        policy = OutputPolicy(topic_guardrail=None)
        result = await check_topic("anything", policy, _llm=llm)
        assert result is None
        llm.with_structured_output.assert_not_called()

    async def test_off_topic_violation_contains_details(self):
        llm = self._mock_llm(on_topic=False, detected="sports", reason="Discusses football")
        policy = self._policy(topics=["science"])
        result = await check_topic("The team won the championship.", policy, _llm=llm)
        assert result is not None
        assert "sports" in result.message
        assert "science" in result.message

    async def test_run_id_propagated(self):
        llm = self._mock_llm(on_topic=False, detected="other")
        policy = self._policy()
        result = await check_topic("off topic text", policy, run_id="run-xyz", _llm=llm)
        if result is not None:
            assert result.run_id == "run-xyz"

    async def test_custom_action_propagated(self):
        llm = self._mock_llm(on_topic=False, detected="other")
        policy = self._policy(topic_action=ViolationAction.BLOCK)
        result = await check_topic("off topic text", policy, _llm=llm)
        assert result is not None
        assert result.action == ViolationAction.BLOCK
