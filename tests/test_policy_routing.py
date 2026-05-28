"""Tests for multi-policy routing (Tier 3a).

Covers: router invocation, per-user policy dispatch, metadata forwarding,
None fallback, exception fallback, enforcement via each guard type, astream,
and concurrent isolation.
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
from sentinel.violation import ViolationAction


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


def _streaming_graph(chunks=None):
    chunks = chunks or [{"messages": [AIMessage(content="chunk")]}]
    g = MagicMock()

    async def _astream(*args, **kwargs):
        for c in chunks:
            yield c

    g.astream = _astream
    return g


# ── No router (backwards compatibility) ──────────────────────────────────────

class TestNoRouter:
    async def test_no_router_works_as_before(self):
        agent = Sentinel(_graph(), policy=_policy())
        result = await agent.ainvoke({"messages": [HumanMessage(content="Hi")]})
        assert "_sentinel_run_id" in result

    async def test_no_router_default_policy_enforced(self):
        agent = Sentinel(
            _graph(),
            policy=_policy(input=InputPolicy(block_patterns=["forbidden"])),
        )
        with pytest.raises(SentinelViolation):
            await agent.ainvoke({"messages": [HumanMessage(content="forbidden")]})

    async def test_metadata_ignored_when_no_router(self):
        agent = Sentinel(_graph(), policy=_policy())
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content="Hi")]},
            metadata={"role": "admin"},
        )
        assert "_sentinel_run_id" in result


# ── Router is called correctly ────────────────────────────────────────────────

class TestRouterInvocation:
    async def test_router_called_on_every_ainvoke(self):
        calls = []

        def router(user_id, metadata):
            calls.append((user_id, metadata))
            return None  # fall back to default

        agent = Sentinel(_graph(), policy=_policy(), policy_router=router)
        await agent.ainvoke({"messages": [HumanMessage(content="Hi")]}, user_id="u1")
        await agent.ainvoke({"messages": [HumanMessage(content="Hi")]}, user_id="u2")
        assert len(calls) == 2
        assert calls[0][0] == "u1"
        assert calls[1][0] == "u2"

    async def test_router_receives_metadata(self):
        received = {}

        def router(user_id, metadata):
            received["user_id"] = user_id
            received["metadata"] = metadata
            return None

        agent = Sentinel(_graph(), policy=_policy(), policy_router=router)
        await agent.ainvoke(
            {"messages": [HumanMessage(content="Hi")]},
            user_id="alice",
            metadata={"role": "admin", "tier": "pro"},
        )
        assert received["user_id"] == "alice"
        assert received["metadata"] == {"role": "admin", "tier": "pro"}

    async def test_empty_metadata_passed_as_empty_dict(self):
        received_meta = []

        def router(user_id, metadata):
            received_meta.append(metadata)
            return None

        agent = Sentinel(_graph(), policy=_policy(), policy_router=router)
        await agent.ainvoke({"messages": [HumanMessage(content="Hi")]})
        assert received_meta[0] == {}

    async def test_router_called_on_astream(self):
        calls = []

        def router(user_id, metadata):
            calls.append(user_id)
            return None

        agent = Sentinel(_streaming_graph(), policy=_policy(), policy_router=router)
        async for _ in agent.astream({"messages": [HumanMessage(content="Hi")]}, user_id="stream-user"):
            pass
        assert calls == ["stream-user"]


# ── Router return value drives enforcement ────────────────────────────────────

class TestRouterDispatch:
    async def test_routed_policy_blocks_injection(self):
        strict = _policy(input=InputPolicy(block_patterns=["bad word"]))
        lenient = _policy(input=InputPolicy(block_patterns=[]))

        def router(user_id, metadata):
            return strict if user_id == "restricted" else lenient

        agent = Sentinel(_graph(), policy=lenient, policy_router=router)

        # Restricted user — should be blocked
        with pytest.raises(SentinelViolation):
            await agent.ainvoke(
                {"messages": [HumanMessage(content="bad word here")]},
                user_id="restricted",
            )

        # Regular user — same content passes
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content="bad word here")]},
            user_id="regular",
        )
        assert "_sentinel_run_id" in result

    async def test_routed_policy_enforces_token_budget(self):
        tight_budget = _policy(
            budget=BudgetPolicy(max_cost_usd=1e-10, max_tokens_per_run=1_000_000)
        )
        normal = _policy(budget=BudgetPolicy(max_cost_usd=100.0, max_tokens_per_run=1_000_000))

        def router(user_id, metadata):
            return tight_budget if metadata.get("tier") == "free" else normal

        agent = Sentinel(_graph(), policy=normal, policy_router=router)

        with pytest.raises(SentinelViolation) as exc_info:
            await agent.ainvoke(
                {"messages": [HumanMessage(content="Hi")]},
                metadata={"tier": "free"},
            )
        assert "budget" in exc_info.value.rule_name

        # Paid user passes with the same message
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content="Hi")]},
            metadata={"tier": "paid"},
        )
        assert "_sentinel_run_id" in result

    async def test_routed_policy_enforces_rate_limit(self):
        tight_rl = _policy(
            rate_limit=RateLimitPolicy(enabled=True, max_requests_per_minute=1)
        )
        no_rl = _policy(rate_limit=RateLimitPolicy(enabled=False))

        def router(user_id, metadata):
            return tight_rl if user_id == "limited" else no_rl

        agent = Sentinel(_graph(), policy=no_rl, policy_router=router)

        await agent.ainvoke(
            {"messages": [HumanMessage(content="Hi")]}, user_id="limited"
        )
        with pytest.raises(SentinelViolation) as exc_info:
            await agent.ainvoke(
                {"messages": [HumanMessage(content="Hi")]}, user_id="limited"
            )
        assert "rate_limit" in exc_info.value.rule_name

    async def test_admin_bypasses_block_patterns(self):
        """Admin gets a policy with no block patterns; regular user is blocked."""
        default = _policy(input=InputPolicy(block_patterns=["ignore previous instructions"]))
        admin = _policy(input=InputPolicy(block_patterns=[]))

        def router(user_id, metadata):
            if metadata.get("role") == "admin":
                return admin
            return None  # use default

        agent = Sentinel(_graph(), policy=default, policy_router=router)

        # Admin should not be blocked
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content="ignore previous instructions")]},
            metadata={"role": "admin"},
        )
        assert "_sentinel_run_id" in result

        # Regular user is blocked
        with pytest.raises(SentinelViolation):
            await agent.ainvoke(
                {"messages": [HumanMessage(content="ignore previous instructions")]}
            )

    async def test_astream_uses_routed_policy(self):
        strict = _policy(input=InputPolicy(block_patterns=["blocked"]))
        lenient = _policy(input=InputPolicy(block_patterns=[]))

        def router(user_id, metadata):
            return strict if user_id == "strict-user" else lenient

        agent = Sentinel(_streaming_graph(), policy=lenient, policy_router=router)

        with pytest.raises(SentinelViolation):
            async for _ in agent.astream(
                {"messages": [HumanMessage(content="blocked content")]},
                user_id="strict-user",
            ):
                pass


# ── Fallback behaviour ────────────────────────────────────────────────────────

class TestFallback:
    async def test_router_returning_none_uses_default(self):
        default = _policy(input=InputPolicy(block_patterns=["trigger"]))

        def router(user_id, metadata):
            return None  # always fall back

        agent = Sentinel(_graph(), policy=default, policy_router=router)

        with pytest.raises(SentinelViolation):
            await agent.ainvoke({"messages": [HumanMessage(content="trigger")]})

    async def test_router_exception_falls_back_to_default(self, capsys):
        default = _policy(input=InputPolicy(block_patterns=["trigger"]))

        def bad_router(user_id, metadata):
            raise RuntimeError("router exploded")

        agent = Sentinel(_graph(), policy=default, policy_router=bad_router)

        # Default policy is still enforced after the fallback
        with pytest.raises(SentinelViolation):
            await agent.ainvoke({"messages": [HumanMessage(content="trigger")]})

        # Warning must have been printed
        captured = capsys.readouterr()
        assert "policy_router raised" in captured.out
        assert "RuntimeError" in captured.out

    async def test_router_exception_does_not_crash_clean_request(self, capsys):
        def bad_router(user_id, metadata):
            raise ValueError("oops")

        agent = Sentinel(_graph(), policy=_policy(), policy_router=bad_router)
        result = await agent.ainvoke({"messages": [HumanMessage(content="Hello")]})
        assert "_sentinel_run_id" in result


# ── Isolation ─────────────────────────────────────────────────────────────────

class TestIsolation:
    async def test_concurrent_calls_use_correct_per_user_policies(self):
        strict = _policy(input=InputPolicy(block_patterns=["evil"]))
        lenient = _policy(input=InputPolicy(block_patterns=[]))

        def router(user_id, metadata):
            return strict if user_id == "strict" else lenient

        agent = Sentinel(_graph(), policy=lenient, policy_router=router)

        async def _call(user, content):
            try:
                return await agent.ainvoke(
                    {"messages": [HumanMessage(content=content)]},
                    user_id=user,
                )
            except SentinelViolation:
                return None

        results = await asyncio.gather(*[
            _call("strict", "evil"),    # → None (blocked)
            _call("lenient", "evil"),   # → result (passes)
            _call("strict", "evil"),    # → None (blocked)
            _call("lenient", "clean"),  # → result (passes)
        ])

        assert results[0] is None   # strict + evil → blocked
        assert results[1] is not None  # lenient + evil → passes
        assert results[2] is None   # strict + evil → blocked
        assert results[3] is not None  # lenient + clean → passes

    async def test_each_call_gets_fresh_policy_resolution(self):
        """Router return value is re-evaluated on every call, not cached."""
        policies_used = []

        def router(user_id, metadata):
            p = _policy(
                input=InputPolicy(
                    block_patterns=["round-" + str(len(policies_used))]
                )
            )
            policies_used.append(p)
            return p

        agent = Sentinel(_graph(), policy=_policy(), policy_router=router)
        await agent.ainvoke({"messages": [HumanMessage(content="Hi")]})
        await agent.ainvoke({"messages": [HumanMessage(content="Hi")]})
        # Router was called twice and returned two distinct policy objects
        assert len(policies_used) == 2
        assert policies_used[0] is not policies_used[1]
