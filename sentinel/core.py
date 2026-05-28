from __future__ import annotations

import asyncio
import threading
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Callable, Optional

from langchain_core.runnables import RunnableConfig

from sentinel.async_guards.hallucination import check as check_hallucination
from sentinel.async_guards.output_validator import validate as validate_output
from sentinel.async_guards.schema_validator import validate as validate_schema
from sentinel.async_guards.topic_guardrail import check as check_topic
from sentinel.callbacks import SentinelCallbackHandler
from sentinel.policy import SentinelPolicy, load_policy
from sentinel.sync_guards.budget_gate import BudgetTracker
from sentinel.sync_guards.circuit_breaker import CircuitBreakerState
from sentinel.sync_guards.input_validator import validate as validate_input
from sentinel.sync_guards.rate_limiter import RateLimiter
from sentinel.stores.base import ViolationStore
from sentinel.telemetry import emit_violation
from sentinel import metrics as _metrics
from sentinel.violation import ViolationAction, ViolationLog
from sentinel.watcher import PolicyWatcher

_BLOCKING_ACTIONS = {ViolationAction.BLOCK, ViolationAction.ABORT}


PolicyRouter = Callable[[str, dict[str, Any]], Optional[SentinelPolicy]]


class Sentinel:
    def __init__(
        self,
        graph: Any,
        policy: str | Path | SentinelPolicy,
        policy_router: Optional[PolicyRouter] = None,
        shadow_router: Optional[PolicyRouter] = None,
        store: Optional[ViolationStore] = None,
    ) -> None:
        self._policy_lock = threading.Lock()
        self._watcher: Optional[PolicyWatcher] = None

        if isinstance(policy, (str, Path)):
            self._policy: SentinelPolicy = load_policy(policy)
            self._policy_path: Optional[Path] = Path(policy)
            self._watcher = PolicyWatcher(self._policy_path, self._on_policy_reload)
            self._watcher.start()
        else:
            self._policy = policy
            self._policy_path = None

        self.graph = graph
        self._policy_router = policy_router
        self._shadow_router = shadow_router
        self._store = store  # None = no external persistence (default)
        self._budget_tracker = BudgetTracker()
        self._circuit_breaker = CircuitBreakerState()
        self._rate_limiter = RateLimiter()
        self._violation_log: dict[str, list[ViolationLog]] = {}
        # Kept alive so GC doesn't drop fire-and-forget tasks before they finish
        self._background_tasks: set[asyncio.Task] = set()

    @property
    def policy(self) -> SentinelPolicy:
        with self._policy_lock:
            return self._policy

    @policy.setter
    def policy(self, new_policy: SentinelPolicy) -> None:
        with self._policy_lock:
            self._policy = new_policy

    def _on_policy_reload(self, new_policy: SentinelPolicy) -> None:
        self.policy = new_policy

    def stop_watcher(self) -> None:
        if self._watcher:
            self._watcher.stop()
            self._watcher = None

    def _resolve_policy(self, user_id: str, metadata: dict[str, Any]) -> SentinelPolicy:
        """Return the effective policy for this call.

        If a policy_router is configured it is called first; its return value is
        used when non-None.  Exceptions from the router are caught, logged, and
        fall through to the default policy so a buggy router never silences the
        agent.
        """
        if self._policy_router is not None:
            try:
                routed = self._policy_router(user_id, metadata)
                if routed is not None:
                    return routed
            except Exception as exc:
                print(
                    f"[sentinel] policy_router raised ({type(exc).__name__}): {exc}"
                    " — falling back to default policy",
                    flush=True,
                )
        return self.policy  # thread-safe property

    def _resolve_shadow_policy(
        self, user_id: str, metadata: dict[str, Any]
    ) -> Optional[SentinelPolicy]:
        """Return the shadow policy for this call, or None if shadow mode is off.

        Exceptions from the shadow_router are caught and logged; shadow mode is
        simply disabled for that call rather than crashing the request.
        """
        if self._shadow_router is None:
            return None
        try:
            return self._shadow_router(user_id, metadata)
        except Exception as exc:
            print(
                f"[sentinel] shadow_router raised ({type(exc).__name__}): {exc}"
                " — shadow mode disabled for this call",
                flush=True,
            )
            return None

    def _run_shadow_sync_guards(
        self,
        input_dict: dict[str, Any],
        run_id: str,
        shadow_policy: SentinelPolicy,
        user_id: str = "global",
    ) -> None:
        """Run sync guards under shadow_policy — record violations but never raise.

        All guards use peek / non-committing variants so shadow observation never
        consumes real budget or rate-limit quota.
        """
        combined_input = self._collect_input_texts(input_dict)
        estimated_tokens = self._estimate_input_tokens(input_dict)

        candidates: list = [
            validate_input(combined_input, shadow_policy.input, run_id=run_id, node_name="input"),
            self._budget_tracker.peek(
                run_id=run_id,
                estimated_tokens=estimated_tokens,
                model=shadow_policy.service,
                policy=shadow_policy.budget,
                node_name="input",
            ),
            self._rate_limiter.peek(
                user_id=user_id,
                tokens=estimated_tokens,
                policy=shadow_policy.rate_limit,
                run_id=run_id,
                node_name="input",
            ),
        ]
        for v in candidates:
            if v is not None:
                self._record_violation(run_id, v.as_shadow())

    def _record_violation(self, run_id: str, log: ViolationLog) -> None:
        self._violation_log.setdefault(run_id, []).append(log)
        # Snapshot both values under one lock acquisition to avoid stale reads
        # during a concurrent hot-reload.
        with self._policy_lock:
            endpoint = self._policy.audit.tracely_endpoint
            service_name = self._policy.service
        # Prometheus counter — no-op when prometheus_client not installed
        _metrics.record_violation(log, service=service_name)
        if endpoint:
            # Blocking violations export synchronously — they raise immediately after
            # this call, so a daemon thread would be killed before it could flush.
            emit_violation(
                log,
                tracely_endpoint=endpoint,
                service_name=service_name,
                blocking=log.action in _BLOCKING_ACTIONS,
            )
        # Write-through to persistent store — fire-and-forget via background task
        if self._store is not None:
            try:
                loop = asyncio.get_running_loop()
                task = loop.create_task(self._store.save(run_id, log))
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
            except RuntimeError:
                pass  # no running event loop — should not happen in normal use

    def _estimate_input_tokens(self, input_dict: dict[str, Any]) -> int:
        total = 0
        for v in input_dict.values():
            if isinstance(v, str):
                total += max(len(v.split()), 1)
            elif isinstance(v, list):
                for item in v:
                    if hasattr(item, "content"):
                        total += max(len(str(item.content).split()), 1)
                    elif isinstance(item, str):
                        total += max(len(item.split()), 1)
        return max(total, 1)

    def _collect_input_texts(self, input_dict: dict[str, Any]) -> str:
        texts: list[str] = []
        for val in input_dict.values():
            if isinstance(val, str):
                texts.append(val)
            elif isinstance(val, list):
                for item in val:
                    if hasattr(item, "content") and isinstance(item.content, str):
                        texts.append(item.content)
                    elif isinstance(item, str):
                        texts.append(item)
        return " ".join(texts)

    def _build_config(self, run_id: str, policy: SentinelPolicy) -> RunnableConfig:
        handler = SentinelCallbackHandler(
            run_id=run_id,
            policy=policy,
            budget_tracker=self._budget_tracker,
            circuit_breaker=self._circuit_breaker,
            record_violation_fn=self._record_violation,
        )
        return RunnableConfig(callbacks=[handler])

    def _run_sync_guards(
        self,
        input_dict: dict[str, Any],
        run_id: str,
        policy: SentinelPolicy,
        user_id: str = "global",
    ) -> None:
        """Run all pre-graph guards; raise SentinelViolation if any block."""
        combined_input = self._collect_input_texts(input_dict)

        # (a) Input validator — patterns, token limit, PII, semantic injection
        input_violation = validate_input(
            combined_input, policy.input, run_id=run_id, node_name="input"
        )
        if input_violation:
            self._record_violation(run_id, input_violation)
            if input_violation.action in _BLOCKING_ACTIONS:
                raise input_violation.to_sentinel_violation()

        # (b) Budget pre-flight — peek only; callbacks commit actual tokens per LLM call
        estimated_tokens = self._estimate_input_tokens(input_dict)
        budget_violation = self._budget_tracker.peek(
            run_id=run_id,
            estimated_tokens=estimated_tokens,
            model=policy.service,
            policy=policy.budget,
            node_name="input",
        )
        if budget_violation:
            self._record_violation(run_id, budget_violation)
            if budget_violation.action in _BLOCKING_ACTIONS:
                raise budget_violation.to_sentinel_violation()

        # (c) Cross-run rate limiter — keyed by user_id
        rate_violation = self._rate_limiter.check(
            user_id=user_id,
            tokens=estimated_tokens,
            policy=policy.rate_limit,
            run_id=run_id,
            node_name="input",
        )
        if rate_violation:
            self._record_violation(run_id, rate_violation)
            # Update Prometheus rate-usage gauges
            usage = self._rate_limiter.get_usage(user_id)
            _metrics.update_rate_usage(
                user_id=user_id,
                requests_last_minute=usage["requests_last_minute"],
                tokens_last_hour=usage["tokens_last_hour"],
                service=policy.service,
            )
            if rate_violation.action in _BLOCKING_ACTIONS:
                raise rate_violation.to_sentinel_violation()

    def _schedule_async_guards(
        self,
        run_id: str,
        output_text: str,
        combined_input: str,
        policy: SentinelPolicy,
        shadow: bool = False,
    ) -> None:
        if not output_text:
            return
        task = asyncio.get_running_loop().create_task(
            self._run_async_guards(
                run_id=run_id,
                output_text=output_text,
                original_input=combined_input,
                policy=policy,
                shadow=shadow,
            )
        )
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def ainvoke(
        self,
        input_dict: dict[str, Any],
        *,
        user_id: str = "global",
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        run_id = str(uuid.uuid4())
        self._violation_log[run_id] = []

        # Resolve policy once — router takes precedence over the default
        policy = self._resolve_policy(user_id, metadata or {})
        combined_input = self._collect_input_texts(input_dict)

        self._run_sync_guards(input_dict, run_id, policy, user_id=user_id)

        # Shadow sync guards — observe without enforcing
        shadow_policy = self._resolve_shadow_policy(user_id, metadata or {})
        if shadow_policy is not None:
            self._run_shadow_sync_guards(input_dict, run_id, shadow_policy, user_id=user_id)

        config = self._build_config(run_id, policy)
        result = await self.graph.ainvoke(input_dict, config)

        output_text = _extract_output_text(result)
        self._schedule_async_guards(run_id, output_text, combined_input, policy)
        if shadow_policy is not None:
            self._schedule_async_guards(run_id, output_text, combined_input, shadow_policy, shadow=True)

        # Update budget Prometheus gauges after the call
        usage = self._budget_tracker.get_usage(run_id)
        _metrics.update_budget(run_id, int(usage["tokens"]), usage["cost_usd"], policy.service)

        # Shallow-copy to avoid mutating the graph's returned object in-place.
        # Multiple sequential/concurrent calls would otherwise share the same dict
        # and overwrite each other's _sentinel_run_id.
        if isinstance(result, dict):
            result = {**result, "_sentinel_run_id": run_id}

        return result

    async def astream(
        self,
        input_dict: dict[str, Any],
        *,
        user_id: str = "global",
        metadata: Optional[dict[str, Any]] = None,
    ) -> AsyncIterator[dict[str, Any]]:
        run_id = str(uuid.uuid4())
        self._violation_log[run_id] = []

        policy = self._resolve_policy(user_id, metadata or {})
        combined_input = self._collect_input_texts(input_dict)

        self._run_sync_guards(input_dict, run_id, policy, user_id=user_id)

        shadow_policy = self._resolve_shadow_policy(user_id, metadata or {})
        if shadow_policy is not None:
            self._run_shadow_sync_guards(input_dict, run_id, shadow_policy, user_id=user_id)

        config = self._build_config(run_id, policy)
        accumulated: list[str] = []

        async for chunk in self.graph.astream(input_dict, config):
            chunk_text = _extract_output_text(chunk)
            if chunk_text:
                accumulated.append(chunk_text)
            yield {**chunk, "_sentinel_run_id": run_id} if isinstance(chunk, dict) else chunk

        full_output = " ".join(accumulated)
        self._schedule_async_guards(run_id, full_output, combined_input, policy)
        if shadow_policy is not None:
            self._schedule_async_guards(run_id, full_output, combined_input, shadow_policy, shadow=True)

        usage = self._budget_tracker.get_usage(run_id)
        _metrics.update_budget(run_id, int(usage["tokens"]), usage["cost_usd"], policy.service)

    async def _run_async_guards(
        self,
        run_id: str,
        output_text: str,
        original_input: str = "",
        policy: Optional[SentinelPolicy] = None,
        shadow: bool = False,
    ) -> None:
        if policy is None:
            policy = self.policy
        tasks = [
            validate_output(output_text, policy.output, run_id=run_id),
            check_hallucination(
                query=original_input,
                response=output_text,
                context=original_input,
                policy=policy.hallucination,
                run_id=run_id,
            ),
            check_topic(output_text, policy.output, run_id=run_id),
            validate_schema(output_text, policy.output, run_id=run_id),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, ViolationLog):
                self._record_violation(run_id, r.as_shadow() if shadow else r)
            elif isinstance(r, Exception):
                # Log but never crash — async guards must not affect the main path
                print(f"[sentinel] async guard error ({type(r).__name__}): {r}", flush=True)

    async def get_violations(
        self,
        run_id: str,
        *,
        include_shadow: bool = True,
    ) -> list[ViolationLog]:
        """Return violations for `run_id`.

        Reads from the in-memory dict for runs in the current session.  For
        run IDs not present (e.g. from a previous process), falls back to the
        configured persistent store if one is set.

        Pass include_shadow=False to see only enforced (non-shadow) violations.
        """
        if run_id in self._violation_log:
            logs = self._violation_log[run_id]
            if include_shadow:
                return list(logs)
            return [v for v in logs if not v.shadow]

        if self._store is not None:
            return await self._store.get(run_id, include_shadow=include_shadow)

        return []


def _extract_output_text(result: Any) -> str:
    if not isinstance(result, dict):
        return str(result)[:2000]
    for key in ("messages", "output", "response", "content"):
        val = result.get(key)
        if val is None:
            continue
        if isinstance(val, str):
            return val
        if isinstance(val, list) and val:
            last = val[-1]
            if hasattr(last, "content"):
                return str(last.content)
            if isinstance(last, str):
                return last
    # Fallback: stringify but cap to avoid huge spans
    return str(result)[:2000]
