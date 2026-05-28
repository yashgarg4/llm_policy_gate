"""Per-node LLM interception via LangChain AsyncCallbackHandler.

Commits actual token usage to BudgetTracker and fires CircuitBreakerState
checks after each LLM call within the wrapped graph.
"""
from __future__ import annotations

from typing import Any, Union

from langchain_core.callbacks.base import AsyncCallbackHandler
from langchain_core.outputs import LLMResult

from sentinel.policy import SentinelPolicy
from sentinel.sync_guards.budget_gate import BudgetTracker
from sentinel.sync_guards.circuit_breaker import CircuitBreakerState
from sentinel.violation import ViolationAction, ViolationLog

_BLOCKING_ACTIONS = {ViolationAction.BLOCK, ViolationAction.ABORT}


class SentinelCallbackHandler(AsyncCallbackHandler):
    """Fires on every LLM call within the wrapped graph.

    - on_chat_model_start / on_llm_start: records current node name
    - on_llm_end: commits actual token usage and checks circuit breaker
    - on_llm_error: records LLM retry so circuit breaker can trip on floods
    """

    raise_error = True  # Propagates SentinelViolation through LangChain's callback chain

    def __init__(
        self,
        run_id: str,
        policy: SentinelPolicy,
        budget_tracker: BudgetTracker,
        circuit_breaker: CircuitBreakerState,
        record_violation_fn,  # (run_id: str, log: ViolationLog) -> None
    ) -> None:
        super().__init__()
        self._sentinel_run_id = run_id
        self._policy = policy
        self._budget_tracker = budget_tracker
        self._circuit_breaker = circuit_breaker
        self._record_violation = record_violation_fn
        self._current_node: str = "llm"

    # ── LLM lifecycle ─────────────────────────────────────────────────────────

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list,
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        """Called by chat models (Gemini, GPT-4, Claude, etc.) before generation."""
        self._current_node = (serialized or {}).get("name", "llm")

    async def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        """Called by text-completion LLMs before generation."""
        self._current_node = (serialized or {}).get("name", "llm")

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        """Commit actual token usage and check budget + circuit breaker."""
        actual_tokens = _extract_token_count(response)
        if actual_tokens > 0:
            budget_violation = self._budget_tracker.check(
                run_id=self._sentinel_run_id,
                estimated_tokens=actual_tokens,
                model=self._policy.service,
                policy=self._policy.budget,
                node_name=self._current_node,
            )
            if budget_violation:
                self._record_violation(self._sentinel_run_id, budget_violation)
                if budget_violation.action in _BLOCKING_ACTIONS:
                    raise budget_violation.to_sentinel_violation()

        cb_violation = self._circuit_breaker.check(
            run_id=self._sentinel_run_id,
            node_name=self._current_node,
            policy=self._policy.circuit_breaker,
        )
        if cb_violation:
            self._record_violation(self._sentinel_run_id, cb_violation)
            if cb_violation.action in _BLOCKING_ACTIONS:
                raise cb_violation.to_sentinel_violation()

    async def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        """Record an LLM failure so the circuit breaker can trip on retry floods."""
        self._circuit_breaker.record_retry(self._sentinel_run_id)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_token_count(response: LLMResult) -> int:
    """Return total tokens from an LLMResult, or 0 if unavailable."""
    # OpenAI / Anthropic style — llm_output["token_usage"] or ["usage"]
    if response.llm_output:
        usage = (
            response.llm_output.get("token_usage")
            or response.llm_output.get("usage")
            or {}
        )
        total = usage.get("total_tokens") or (
            usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        )
        if total:
            return int(total)

    # Google / Gemini style — usage_metadata on individual ChatGeneration messages
    for gen_list in response.generations:
        for gen in gen_list:
            if hasattr(gen, "message"):
                meta = getattr(gen.message, "usage_metadata", None) or {}
                total = meta.get("total_tokens") or (
                    meta.get("input_tokens", 0) + meta.get("output_tokens", 0)
                )
                if total:
                    return int(total)

    return 0
