from __future__ import annotations

import threading
from collections import defaultdict
from typing import Optional

from sentinel.policy import BudgetPolicy
from sentinel.violation import ViolationLog, ViolationSeverity

# Cost per 1K input tokens by model — longest-prefix match wins
_COST_PER_1K_INPUT: dict[str, float] = {
    "gemini-2.5-pro": 0.00125,
    "gemini-2.0-flash": 0.0001,
    "gemini-1.5-pro": 0.00125,
    "gemini-1.5-flash": 0.000075,
    "gpt-4o-mini": 0.00015,
    "gpt-4o": 0.005,
    "gpt-4": 0.03,
    "gpt-3.5": 0.0005,
    "claude-3-opus": 0.015,
    "claude-3-sonnet": 0.003,
    "claude-3-haiku": 0.00025,
}

_DEFAULT_COST_PER_1K = 0.001


def _estimate_cost(tokens: int, model: str) -> float:
    model_lower = model.lower()
    # Exact match first, then longest prefix — avoids gpt-4 matching gpt-4o
    if model_lower in _COST_PER_1K_INPUT:
        return (tokens / 1000) * _COST_PER_1K_INPUT[model_lower]
    best_prefix = max(
        (p for p in _COST_PER_1K_INPUT if model_lower.startswith(p)),
        key=len,
        default=None,
    )
    cost = _COST_PER_1K_INPUT[best_prefix] if best_prefix else _DEFAULT_COST_PER_1K
    return (tokens / 1000) * cost


class BudgetTracker:
    def __init__(self) -> None:
        self._run_tokens: dict[str, int] = defaultdict(int)
        self._run_cost: dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()

    def reset(self, run_id: str) -> None:
        with self._lock:
            self._run_tokens.pop(run_id, None)
            self._run_cost.pop(run_id, None)

    def check(
        self,
        run_id: str,
        estimated_tokens: int,
        model: str,
        policy: BudgetPolicy,
        node_name: str = "",
    ) -> Optional[ViolationLog]:
        with self._lock:
            projected_tokens = self._run_tokens[run_id] + estimated_tokens
            projected_cost = self._run_cost[run_id] + _estimate_cost(estimated_tokens, model)

            if projected_tokens > policy.max_tokens_per_run:
                return ViolationLog(
                    rule_name="budget.max_tokens_per_run",
                    action=policy.action,
                    severity=ViolationSeverity.HIGH,
                    message=f"Token budget exceeded: {projected_tokens} > {policy.max_tokens_per_run}",
                    offending_content=f"run_id={run_id}, model={model}",
                    run_id=run_id,
                    node_name=node_name,
                )

            if projected_cost > policy.max_cost_usd:
                return ViolationLog(
                    rule_name="budget.max_cost_usd",
                    action=policy.action,
                    severity=ViolationSeverity.HIGH,
                    message=f"Cost budget exceeded: ${projected_cost:.4f} > ${policy.max_cost_usd:.4f}",
                    offending_content=f"run_id={run_id}, model={model}",
                    run_id=run_id,
                    node_name=node_name,
                )

            # Commit inside the lock — only on success
            self._run_tokens[run_id] = projected_tokens
            self._run_cost[run_id] = projected_cost
            return None

    def peek(
        self,
        run_id: str,
        estimated_tokens: int,
        model: str,
        policy: BudgetPolicy,
        node_name: str = "",
    ) -> Optional[ViolationLog]:
        """Read-only budget check — does not commit tokens. Used for pre-flight validation."""
        with self._lock:
            projected_tokens = self._run_tokens[run_id] + estimated_tokens
            projected_cost = self._run_cost[run_id] + _estimate_cost(estimated_tokens, model)

        if projected_tokens > policy.max_tokens_per_run:
            return ViolationLog(
                rule_name="budget.max_tokens_per_run",
                action=policy.action,
                severity=ViolationSeverity.HIGH,
                message=f"Token budget exceeded: {projected_tokens} > {policy.max_tokens_per_run}",
                offending_content=f"run_id={run_id}, model={model}",
                run_id=run_id,
                node_name=node_name,
            )

        if projected_cost > policy.max_cost_usd:
            return ViolationLog(
                rule_name="budget.max_cost_usd",
                action=policy.action,
                severity=ViolationSeverity.HIGH,
                message=f"Cost budget exceeded: ${projected_cost:.4f} > ${policy.max_cost_usd:.4f}",
                offending_content=f"run_id={run_id}, model={model}",
                run_id=run_id,
                node_name=node_name,
            )

        return None

    def adjust(self, run_id: str, delta_tokens: int, model: str = "") -> None:
        """Add delta_tokens to the committed count after actual LLM usage is known."""
        if delta_tokens == 0:
            return
        with self._lock:
            self._run_tokens[run_id] = max(0, self._run_tokens[run_id] + delta_tokens)
            if model and delta_tokens > 0:
                self._run_cost[run_id] = max(
                    0.0,
                    self._run_cost[run_id] + _estimate_cost(delta_tokens, model),
                )

    def get_usage(self, run_id: str) -> dict[str, float]:
        with self._lock:
            return {
                "tokens": self._run_tokens[run_id],
                "cost_usd": self._run_cost[run_id],
            }
