"""Tests for SentinelCallbackHandler — per-node budget commit and circuit breaker."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.messages import AIMessage

from sentinel.callbacks import SentinelCallbackHandler, _extract_token_count
from sentinel.policy import (
    BudgetPolicy,
    CircuitBreakerPolicy,
    HallucinationPolicy,
    AuditPolicy,
    InputPolicy,
    OutputPolicy,
    SentinelPolicy,
)
from sentinel.sync_guards.budget_gate import BudgetTracker
from sentinel.sync_guards.circuit_breaker import CircuitBreakerState
from sentinel.violation import ViolationAction, ViolationLog, ViolationSeverity
from sentinel import SentinelViolation


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _policy(**overrides) -> SentinelPolicy:
    defaults = dict(
        input=InputPolicy(block_patterns=[], max_tokens=4096),
        budget=BudgetPolicy(max_cost_usd=10.0, max_tokens_per_run=100_000),
        circuit_breaker=CircuitBreakerPolicy(max_node_repeats=10, max_retries=10),
        output=OutputPolicy(toxicity_check=False),
        hallucination=HallucinationPolicy(enabled=False),
        audit=AuditPolicy(log_all=True, tracely_endpoint=None),
    )
    defaults.update(overrides)
    return SentinelPolicy(**defaults)


def _make_handler(
    run_id: str = "run-test",
    policy: SentinelPolicy | None = None,
    budget_tracker: BudgetTracker | None = None,
    circuit_breaker: CircuitBreakerState | None = None,
    violations: list | None = None,
) -> tuple[SentinelCallbackHandler, list[ViolationLog]]:
    policy = policy or _policy()
    tracker = budget_tracker or BudgetTracker()
    cb = circuit_breaker or CircuitBreakerState()
    log: list[ViolationLog] = violations if violations is not None else []

    def _record(rid: str, v: ViolationLog) -> None:
        log.append(v)

    handler = SentinelCallbackHandler(
        run_id=run_id,
        policy=policy,
        budget_tracker=tracker,
        circuit_breaker=cb,
        record_violation_fn=_record,
    )
    return handler, log


def _llm_result(total_tokens: int = 0, via: str = "llm_output") -> LLMResult:
    """Build an LLMResult with token counts in either the llm_output or usage_metadata style."""
    if via == "llm_output":
        return LLMResult(
            generations=[[]],
            llm_output={"token_usage": {"total_tokens": total_tokens}},
        )
    # Gemini / usage_metadata style
    msg = AIMessage(content="ok")
    msg.usage_metadata = {"total_tokens": total_tokens}  # type: ignore[attr-defined]
    gen = ChatGeneration(message=msg, text="ok")
    return LLMResult(generations=[[gen]])


# ── _extract_token_count ──────────────────────────────────────────────────────

class TestExtractTokenCount:
    def test_llm_output_token_usage(self):
        r = LLMResult(
            generations=[[]],
            llm_output={"token_usage": {"total_tokens": 42}},
        )
        assert _extract_token_count(r) == 42

    def test_llm_output_usage_key(self):
        r = LLMResult(
            generations=[[]],
            llm_output={"usage": {"total_tokens": 100}},
        )
        assert _extract_token_count(r) == 100

    def test_llm_output_input_plus_output(self):
        r = LLMResult(
            generations=[[]],
            llm_output={"token_usage": {"input_tokens": 30, "output_tokens": 20}},
        )
        assert _extract_token_count(r) == 50

    def test_usage_metadata_on_chat_generation(self):
        msg = AIMessage(content="hi")
        msg.usage_metadata = {"total_tokens": 77}  # type: ignore[attr-defined]
        gen = ChatGeneration(message=msg, text="hi")
        r = LLMResult(generations=[[gen]])
        assert _extract_token_count(r) == 77

    def test_usage_metadata_input_plus_output(self):
        msg = AIMessage(content="hi")
        msg.usage_metadata = {"input_tokens": 10, "output_tokens": 25}  # type: ignore[attr-defined]
        gen = ChatGeneration(message=msg, text="hi")
        r = LLMResult(generations=[[gen]])
        assert _extract_token_count(r) == 35

    def test_no_usage_info_returns_zero(self):
        r = LLMResult(generations=[[]])
        assert _extract_token_count(r) == 0

    def test_empty_llm_output_returns_zero(self):
        r = LLMResult(generations=[[]], llm_output={})
        assert _extract_token_count(r) == 0


# ── on_chat_model_start / on_llm_start ───────────────────────────────────────

class TestNodeNameTracking:
    async def test_chat_model_start_sets_node_name(self):
        handler, _ = _make_handler()
        await handler.on_chat_model_start(
            {"name": "gemini-flash"}, [], run_id="x"
        )
        assert handler._current_node == "gemini-flash"

    async def test_llm_start_sets_node_name(self):
        handler, _ = _make_handler()
        await handler.on_llm_start(
            {"name": "gpt-4o"}, ["prompt"], run_id="x"
        )
        assert handler._current_node == "gpt-4o"

    async def test_missing_name_falls_back_to_llm(self):
        handler, _ = _make_handler()
        await handler.on_chat_model_start({}, [], run_id="x")
        assert handler._current_node == "llm"

    async def test_none_serialized_falls_back_to_llm(self):
        handler, _ = _make_handler()
        await handler.on_chat_model_start(None, [], run_id="x")
        assert handler._current_node == "llm"


# ── on_llm_end — budget commit ────────────────────────────────────────────────

class TestOnLlmEndBudget:
    async def test_actual_tokens_committed_to_tracker(self):
        tracker = BudgetTracker()
        handler, _ = _make_handler(run_id="r1", budget_tracker=tracker)
        result = _llm_result(total_tokens=500)
        await handler.on_llm_end(result, run_id="x")
        assert tracker.get_usage("r1")["tokens"] == 500

    async def test_zero_tokens_skips_commit(self):
        tracker = BudgetTracker()
        handler, _ = _make_handler(run_id="r1", budget_tracker=tracker)
        await handler.on_llm_end(_llm_result(0), run_id="x")
        assert tracker.get_usage("r1")["tokens"] == 0

    async def test_budget_exceeded_raises_sentinel_violation(self):
        policy = _policy(budget=BudgetPolicy(max_cost_usd=10.0, max_tokens_per_run=10))
        handler, violations = _make_handler(policy=policy)
        with pytest.raises(SentinelViolation) as exc_info:
            await handler.on_llm_end(_llm_result(100), run_id="x")
        assert "budget" in exc_info.value.rule_name

    async def test_budget_violation_logged_before_raise(self):
        policy = _policy(budget=BudgetPolicy(max_cost_usd=10.0, max_tokens_per_run=5))
        handler, violations = _make_handler(policy=policy)
        with pytest.raises(SentinelViolation):
            await handler.on_llm_end(_llm_result(100), run_id="x")
        assert any("budget" in v.rule_name for v in violations)

    async def test_budget_flag_action_does_not_raise(self):
        policy = _policy(budget=BudgetPolicy(
            max_cost_usd=10.0, max_tokens_per_run=5, action=ViolationAction.FLAG
        ))
        handler, violations = _make_handler(policy=policy)
        await handler.on_llm_end(_llm_result(100), run_id="x")
        assert any("budget" in v.rule_name for v in violations)

    async def test_multiple_calls_accumulate_tokens(self):
        tracker = BudgetTracker()
        handler, _ = _make_handler(run_id="r-acc", budget_tracker=tracker)
        await handler.on_llm_end(_llm_result(100), run_id="x")
        await handler.on_llm_end(_llm_result(200), run_id="x")
        assert tracker.get_usage("r-acc")["tokens"] == 300


# ── on_llm_end — circuit breaker ─────────────────────────────────────────────

class TestOnLlmEndCircuitBreaker:
    async def test_circuit_breaker_fires_after_max_repeats(self):
        policy = _policy(
            circuit_breaker=CircuitBreakerPolicy(max_node_repeats=2, max_retries=100)
        )
        handler, violations = _make_handler(policy=policy)
        # First two calls are fine
        await handler.on_llm_end(_llm_result(0), run_id="x")
        await handler.on_llm_end(_llm_result(0), run_id="x")
        # Third call trips the breaker
        with pytest.raises(SentinelViolation) as exc_info:
            await handler.on_llm_end(_llm_result(0), run_id="x")
        assert "circuit_breaker" in exc_info.value.rule_name

    async def test_circuit_breaker_violation_logged(self):
        policy = _policy(
            circuit_breaker=CircuitBreakerPolicy(max_node_repeats=1, max_retries=100)
        )
        handler, violations = _make_handler(policy=policy)
        await handler.on_llm_end(_llm_result(0), run_id="x")
        with pytest.raises(SentinelViolation):
            await handler.on_llm_end(_llm_result(0), run_id="x")
        assert any("circuit_breaker" in v.rule_name for v in violations)

    async def test_different_run_ids_have_independent_counts(self):
        policy = _policy(
            circuit_breaker=CircuitBreakerPolicy(max_node_repeats=1, max_retries=100)
        )
        cb = CircuitBreakerState()
        handler_a, _ = _make_handler(run_id="run-a", policy=policy, circuit_breaker=cb)
        handler_b, _ = _make_handler(run_id="run-b", policy=policy, circuit_breaker=cb)
        # Each handler has its own run_id — neither should trip after 1 call each
        await handler_a.on_llm_end(_llm_result(0), run_id="x")
        await handler_b.on_llm_end(_llm_result(0), run_id="x")
        # No violation raised — passes implicitly


# ── on_llm_error ─────────────────────────────────────────────────────────────

class TestOnLlmError:
    async def test_error_increments_retry_count(self):
        cb = CircuitBreakerState()
        handler, _ = _make_handler(run_id="r-retry", circuit_breaker=cb)
        await handler.on_llm_error(RuntimeError("timeout"), run_id="x")
        assert cb._retry_counts["r-retry"] == 1

    async def test_multiple_errors_accumulate(self):
        cb = CircuitBreakerState()
        handler, _ = _make_handler(run_id="r-multi", circuit_breaker=cb)
        await handler.on_llm_error(RuntimeError("e1"), run_id="x")
        await handler.on_llm_error(RuntimeError("e2"), run_id="x")
        await handler.on_llm_error(RuntimeError("e3"), run_id="x")
        assert cb._retry_counts["r-multi"] == 3

    async def test_error_does_not_raise(self):
        handler, _ = _make_handler()
        try:
            await handler.on_llm_error(ValueError("oops"), run_id="x")
        except Exception as exc:
            pytest.fail(f"on_llm_error raised unexpectedly: {exc}")


# ── raise_error class attribute ───────────────────────────────────────────────

class TestRaiseError:
    def test_raise_error_is_true(self):
        assert SentinelCallbackHandler.raise_error is True
