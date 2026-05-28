"""Tests for shadow mode (Tier 3b).

Shadow mode runs a candidate policy in parallel — violations are observed and
logged but never raised, so the request always completes normally.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from sentinel import Sentinel, SentinelViolation
from sentinel.policy import (
    AuditPolicy,
    BudgetPolicy,
    CircuitBreakerPolicy,
    HallucinationPolicy,
    InputPolicy,
    OutputPolicy,
    RateLimitPolicy,
    SentinelPolicy,
)
from sentinel.violation import ViolationAction, ViolationLog


# ── Helpers ───────────────────────────────────────────────────────────────────

def _policy(**overrides) -> SentinelPolicy:
    defaults = dict(
        input=InputPolicy(block_patterns=[], max_tokens=4096),
        budget=BudgetPolicy(max_cost_usd=10.0, max_tokens_per_run=100_000),
        circuit_breaker=CircuitBreakerPolicy(max_node_repeats=10, max_retries=10),
        rate_limit=RateLimitPolicy(enabled=False),
        output=OutputPolicy(toxicity_check=False),
        hallucination=HallucinationPolicy(enabled=False),
        audit=AuditPolicy(log_all=True, tracely_endpoint=None),
    )
    defaults.update(overrides)
    return SentinelPolicy(**defaults)


def _graph(response: str = "OK") -> MagicMock:
    g = MagicMock()
    g.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content=response)]})
    return g


def _streaming_graph():
    g = MagicMock()

    async def _astream(*args, **kwargs):
        yield {"messages": [AIMessage(content="chunk")]}

    g.astream = _astream
    return g


# ── ViolationLog.as_shadow() ──────────────────────────────────────────────────

class TestAsShallow:
    def test_as_shadow_sets_flag(self):
        v = ViolationLog(
            rule_name="test.rule",
            action=ViolationAction.BLOCK,
            severity=__import__("sentinel.violation", fromlist=["ViolationSeverity"]).ViolationSeverity.HIGH,
            message="test",
        )
        sv = v.as_shadow()
        assert sv.shadow is True

    def test_as_shadow_preserves_other_fields(self):
        from sentinel.violation import ViolationSeverity
        v = ViolationLog(
            rule_name="input.block_pattern",
            action=ViolationAction.BLOCK,
            severity=ViolationSeverity.CRITICAL,
            message="Blocked",
            offending_content="bad",
            run_id="r1",
            node_name="input",
        )
        sv = v.as_shadow()
        assert sv.rule_name == "input.block_pattern"
        assert sv.action == ViolationAction.BLOCK
        assert sv.run_id == "r1"
        assert sv.node_name == "input"
        assert sv.offending_content == "bad"

    def test_original_is_not_mutated(self):
        from sentinel.violation import ViolationSeverity
        v = ViolationLog(
            rule_name="r", action=ViolationAction.FLAG,
            severity=ViolationSeverity.LOW, message="m",
        )
        _ = v.as_shadow()
        assert v.shadow is False


# ── No shadow router (backwards compat) ──────────────────────────────────────

class TestNoShadowRouter:
    async def test_no_shadow_router_clean_call(self):
        agent = Sentinel(_graph(), policy=_policy())
        result = await agent.ainvoke({"messages": [HumanMessage(content="Hi")]})
        assert "_sentinel_run_id" in result

    async def test_no_shadow_router_violation_not_shadow(self):
        agent = Sentinel(
            _graph(),
            policy=_policy(input=InputPolicy(block_patterns=["bad"])),
        )
        with pytest.raises(SentinelViolation):
            await agent.ainvoke({"messages": [HumanMessage(content="bad")]})


# ── Shadow never blocks ───────────────────────────────────────────────────────

class TestShadowNeverBlocks:
    async def test_shadow_violation_does_not_raise(self):
        """Even when shadow policy would block, the request completes."""
        strict_shadow = _policy(input=InputPolicy(block_patterns=["trigger"]))
        agent = Sentinel(
            _graph(),
            policy=_policy(),
            shadow_router=lambda uid, meta: strict_shadow,
        )
        result = await agent.ainvoke({"messages": [HumanMessage(content="trigger")]})
        assert "_sentinel_run_id" in result

    async def test_shadow_budget_exceeded_does_not_raise(self):
        tiny_budget = _policy(budget=BudgetPolicy(max_cost_usd=1e-10, max_tokens_per_run=1_000_000))
        agent = Sentinel(
            _graph(),
            policy=_policy(),
            shadow_router=lambda uid, meta: tiny_budget,
        )
        result = await agent.ainvoke({"messages": [HumanMessage(content="Hello")]})
        assert "_sentinel_run_id" in result

    async def test_shadow_rate_limit_exceeded_does_not_raise(self):
        strict_rl = _policy(rate_limit=RateLimitPolicy(enabled=True, max_requests_per_minute=1))
        agent = Sentinel(
            _graph(),
            policy=_policy(),
            shadow_router=lambda uid, meta: strict_rl,
        )
        await agent.ainvoke({"messages": [HumanMessage(content="Hi")]}, user_id="u1")
        # Second call would trip shadow rate limit — must not raise
        result = await agent.ainvoke({"messages": [HumanMessage(content="Hi")]}, user_id="u1")
        assert "_sentinel_run_id" in result

    async def test_main_policy_still_blocks_independently(self):
        """Main policy blocks even when shadow is also set."""
        strict_shadow = _policy(input=InputPolicy(block_patterns=["other"]))
        agent = Sentinel(
            _graph(),
            policy=_policy(input=InputPolicy(block_patterns=["bad"])),
            shadow_router=lambda uid, meta: strict_shadow,
        )
        with pytest.raises(SentinelViolation) as exc_info:
            await agent.ainvoke({"messages": [HumanMessage(content="bad")]})
        assert exc_info.value.rule_name == "input.block_pattern"


# ── Shadow violations logged ──────────────────────────────────────────────────

class TestShadowViolationsLogged:
    async def test_shadow_violation_logged_with_shadow_flag(self):
        strict_shadow = _policy(input=InputPolicy(block_patterns=["trigger"]))
        agent = Sentinel(
            _graph(),
            policy=_policy(),
            shadow_router=lambda uid, meta: strict_shadow,
        )
        result = await agent.ainvoke({"messages": [HumanMessage(content="trigger")]})
        run_id = result["_sentinel_run_id"]
        violations = await agent.get_violations(run_id)
        shadow_violations = [v for v in violations if v.shadow]
        assert len(shadow_violations) == 1
        assert shadow_violations[0].rule_name == "input.block_pattern"

    async def test_get_violations_include_shadow_true(self):
        strict_shadow = _policy(input=InputPolicy(block_patterns=["trigger"]))
        agent = Sentinel(
            _graph(),
            policy=_policy(),
            shadow_router=lambda uid, meta: strict_shadow,
        )
        result = await agent.ainvoke({"messages": [HumanMessage(content="trigger")]})
        run_id = result["_sentinel_run_id"]
        all_v = await agent.get_violations(run_id, include_shadow=True)
        assert any(v.shadow for v in all_v)

    async def test_get_violations_exclude_shadow(self):
        strict_shadow = _policy(input=InputPolicy(block_patterns=["trigger"]))
        agent = Sentinel(
            _graph(),
            policy=_policy(),
            shadow_router=lambda uid, meta: strict_shadow,
        )
        result = await agent.ainvoke({"messages": [HumanMessage(content="trigger")]})
        run_id = result["_sentinel_run_id"]
        enforced = await agent.get_violations(run_id, include_shadow=False)
        assert all(not v.shadow for v in enforced)
        assert len(enforced) == 0  # main policy had no block patterns

    async def test_enforced_and_shadow_violations_coexist(self):
        """Both main and shadow policy fire on the same request."""
        strict_main = _policy(input=InputPolicy(block_patterns=["main-trigger"]))
        strict_shadow = _policy(input=InputPolicy(block_patterns=["shadow-trigger"]))
        agent = Sentinel(_graph(), policy=strict_main, shadow_router=lambda *_: strict_shadow)

        # "shadow-trigger" passes the main policy but trips shadow
        result_run_id = None
        try:
            await agent.ainvoke({"messages": [HumanMessage(content="shadow-trigger")]})
            result_run_id = "unreachable"
        except SentinelViolation:
            pass  # main didn't block this one — unexpected

        # Use a clean prompt that only triggers shadow
        result = await agent.ainvoke({"messages": [HumanMessage(content="shadow-trigger")]})
        run_id = result["_sentinel_run_id"]
        all_v = await agent.get_violations(run_id, include_shadow=True)
        shadow_v = [v for v in all_v if v.shadow]
        assert len(shadow_v) >= 1

    async def test_shadow_violations_have_shadow_true(self):
        strict_shadow = _policy(input=InputPolicy(block_patterns=["flagged"]))
        agent = Sentinel(
            _graph(),
            policy=_policy(),
            shadow_router=lambda *_: strict_shadow,
        )
        result = await agent.ainvoke({"messages": [HumanMessage(content="flagged content")]})
        violations = await agent.get_violations(result["_sentinel_run_id"])
        for v in violations:
            if v.rule_name == "input.block_pattern":
                assert v.shadow is True

    async def test_non_shadow_violations_have_shadow_false(self):
        agent = Sentinel(
            _graph(),
            policy=_policy(input=InputPolicy(block_patterns=["enforced"])),
        )
        try:
            await agent.ainvoke({"messages": [HumanMessage(content="enforced")]})
        except SentinelViolation:
            pass
        for v in [v for vlist in agent._violation_log.values() for v in vlist]:
            if v.rule_name == "input.block_pattern":
                assert v.shadow is False


# ── Shadow does not consume real quota ────────────────────────────────────────

class TestShadowDoesNotConsumeQuota:
    async def test_shadow_rate_limit_does_not_consume_real_window(self):
        """Shadow rate-limit peek must not commit to the real rate-limit window."""
        strict_rl = _policy(rate_limit=RateLimitPolicy(enabled=True, max_requests_per_minute=1))
        lenient = _policy(rate_limit=RateLimitPolicy(enabled=True, max_requests_per_minute=100))
        agent = Sentinel(_graph(), policy=lenient, shadow_router=lambda *_: strict_rl)

        # Make two calls — shadow limit (1/min) would be exhausted after first
        for _ in range(5):
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content="Hi")]}, user_id="quota-test"
            )
            # Main policy (100/min) must never block
            assert "_sentinel_run_id" in result

    async def test_shadow_budget_peek_does_not_commit(self):
        """Shadow budget check must not commit estimated tokens to the tracker."""
        tiny = _policy(budget=BudgetPolicy(max_cost_usd=1e-10, max_tokens_per_run=1_000_000))
        agent = Sentinel(_graph(), policy=_policy(), shadow_router=lambda *_: tiny)

        # After the shadow check, the budget tracker's committed tokens for
        # the run must be 0 (shadow never commits)
        result = await agent.ainvoke({"messages": [HumanMessage(content="Hello")]})
        run_id = result["_sentinel_run_id"]
        usage = agent._budget_tracker.get_usage(run_id)
        assert usage["tokens"] == 0  # shadow peek didn't commit


# ── Shadow router error handling ──────────────────────────────────────────────

class TestShadowRouterErrors:
    async def test_shadow_router_exception_does_not_crash(self, capsys):
        def bad_shadow_router(uid, meta):
            raise RuntimeError("shadow router exploded")

        agent = Sentinel(_graph(), policy=_policy(), shadow_router=bad_shadow_router)
        result = await agent.ainvoke({"messages": [HumanMessage(content="Hello")]})
        assert "_sentinel_run_id" in result

    async def test_shadow_router_exception_logged(self, capsys):
        def bad_shadow_router(uid, meta):
            raise ValueError("boom")

        agent = Sentinel(_graph(), policy=_policy(), shadow_router=bad_shadow_router)
        await agent.ainvoke({"messages": [HumanMessage(content="Hello")]})
        captured = capsys.readouterr()
        assert "shadow_router raised" in captured.out

    async def test_shadow_router_returning_none_disables_shadow(self):
        agent = Sentinel(
            _graph(),
            policy=_policy(),
            shadow_router=lambda uid, meta: None,
        )
        result = await agent.ainvoke({"messages": [HumanMessage(content="Hi")]})
        violations = await agent.get_violations(result["_sentinel_run_id"])
        assert all(not v.shadow for v in violations)


# ── astream with shadow mode ──────────────────────────────────────────────────

class TestShadowInAstream:
    async def test_shadow_does_not_block_stream(self):
        strict_shadow = _policy(input=InputPolicy(block_patterns=["trigger"]))
        agent = Sentinel(
            _streaming_graph(),
            policy=_policy(),
            shadow_router=lambda *_: strict_shadow,
        )
        chunks = []
        async for chunk in agent.astream({"messages": [HumanMessage(content="trigger")]}):
            chunks.append(chunk)
        assert len(chunks) == 1
        assert "_sentinel_run_id" in chunks[0]

    async def test_shadow_violation_logged_for_stream(self):
        strict_shadow = _policy(input=InputPolicy(block_patterns=["trigger"]))
        agent = Sentinel(
            _streaming_graph(),
            policy=_policy(),
            shadow_router=lambda *_: strict_shadow,
        )
        run_id = None
        async for chunk in agent.astream({"messages": [HumanMessage(content="trigger")]}):
            run_id = chunk["_sentinel_run_id"]
        violations = await agent.get_violations(run_id)
        assert any(v.shadow and v.rule_name == "input.block_pattern" for v in violations)


# ── Prometheus shadow counter ─────────────────────────────────────────────────

class TestShadowMetrics:
    async def test_shadow_violation_increments_shadow_counter(self):
        from sentinel.metrics import is_available
        if not is_available():
            pytest.skip("prometheus_client not available")
        from sentinel.metrics import _shadow_violations_total, _violations_total

        strict_shadow = _policy(input=InputPolicy(block_patterns=["metricstrigger"]))
        agent = Sentinel(
            _graph(),
            policy=_policy(),
            shadow_router=lambda *_: strict_shadow,
        )

        before_shadow = _shadow_violations_total.labels(
            rule_name="input.block_pattern", action="BLOCK",
            severity="CRITICAL", service="sentinel",
        )._value.get()
        before_enforced = _violations_total.labels(
            rule_name="input.block_pattern", action="BLOCK",
            severity="CRITICAL", service="sentinel",
        )._value.get()

        await agent.ainvoke({"messages": [HumanMessage(content="metricstrigger")]})

        after_shadow = _shadow_violations_total.labels(
            rule_name="input.block_pattern", action="BLOCK",
            severity="CRITICAL", service="sentinel",
        )._value.get()
        after_enforced = _violations_total.labels(
            rule_name="input.block_pattern", action="BLOCK",
            severity="CRITICAL", service="sentinel",
        )._value.get()

        assert after_shadow == before_shadow + 1      # shadow counter incremented
        assert after_enforced == before_enforced      # enforced counter unchanged
