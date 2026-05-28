from __future__ import annotations

from collections import defaultdict
from typing import Optional

from sentinel.policy import CircuitBreakerPolicy, ViolationAction
from sentinel.violation import ViolationLog, ViolationSeverity


class CircuitBreakerState:
    def __init__(self) -> None:
        # per run_id → {node_name: count}
        self._node_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # per run_id → total retry count
        self._retry_counts: dict[str, int] = defaultdict(int)

    def reset(self, run_id: str) -> None:
        self._node_counts.pop(run_id, None)
        self._retry_counts.pop(run_id, None)

    def record_retry(self, run_id: str) -> None:
        self._retry_counts[run_id] += 1

    def check(
        self,
        run_id: str,
        node_name: str,
        policy: CircuitBreakerPolicy,
    ) -> Optional[ViolationLog]:
        self._node_counts[run_id][node_name] += 1
        node_count = self._node_counts[run_id][node_name]
        retry_count = self._retry_counts[run_id]

        if node_count > policy.max_node_repeats:
            return ViolationLog(
                rule_name="circuit_breaker.max_node_repeats",
                action=policy.action,
                severity=ViolationSeverity.CRITICAL,
                message=(
                    f"Node '{node_name}' executed {node_count} times "
                    f"(limit: {policy.max_node_repeats})"
                ),
                offending_content=f"node={node_name}, count={node_count}",
                run_id=run_id,
                node_name=node_name,
            )

        if retry_count > policy.max_retries:
            return ViolationLog(
                rule_name="circuit_breaker.max_retries",
                action=policy.action,
                severity=ViolationSeverity.CRITICAL,
                message=(
                    f"Total retries exceeded: {retry_count} > {policy.max_retries}"
                ),
                offending_content=f"run_id={run_id}, retries={retry_count}",
                run_id=run_id,
                node_name=node_name,
            )

        return None
