"""Tests for the cross-run sliding-window RateLimiter."""
from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from sentinel.policy import RateLimitPolicy
from sentinel.sync_guards.rate_limiter import RateLimiter
from sentinel.violation import ViolationAction


def _policy(**overrides) -> RateLimitPolicy:
    defaults = dict(
        enabled=True,
        max_requests_per_minute=5,
        max_tokens_per_hour=1000,
        action=ViolationAction.BLOCK,
    )
    defaults.update(overrides)
    return RateLimitPolicy(**defaults)


# ── Disabled policy ───────────────────────────────────────────────────────────

class TestDisabled:
    def test_disabled_policy_always_passes(self):
        limiter = RateLimiter()
        policy = _policy(enabled=False)
        # Far exceeds both limits, but policy is disabled
        for _ in range(100):
            v = limiter.check("u1", tokens=99999, policy=policy)
            assert v is None

    def test_enabled_false_default(self):
        assert RateLimitPolicy().enabled is False


# ── Request rate limit ────────────────────────────────────────────────────────

class TestRequestRateLimit:
    def test_within_limit_returns_none(self):
        limiter = RateLimiter()
        policy = _policy(max_requests_per_minute=3)
        for _ in range(3):
            assert limiter.check("u1", tokens=1, policy=policy) is None

    def test_exceeding_limit_returns_violation(self):
        limiter = RateLimiter()
        policy = _policy(max_requests_per_minute=3)
        for _ in range(3):
            limiter.check("u1", tokens=1, policy=policy)
        v = limiter.check("u1", tokens=1, policy=policy)
        assert v is not None
        assert v.rule_name == "rate_limit.requests_per_minute"

    def test_violation_action_propagated(self):
        limiter = RateLimiter()
        policy = _policy(max_requests_per_minute=1, action=ViolationAction.FLAG)
        limiter.check("u1", tokens=1, policy=policy)
        v = limiter.check("u1", tokens=1, policy=policy)
        assert v.action == ViolationAction.FLAG

    def test_different_users_have_independent_counters(self):
        limiter = RateLimiter()
        policy = _policy(max_requests_per_minute=2)
        for _ in range(2):
            limiter.check("alice", tokens=1, policy=policy)
        # Alice is at the limit but Bob is not
        assert limiter.check("alice", tokens=1, policy=policy) is not None
        assert limiter.check("bob", tokens=1, policy=policy) is None

    def test_violation_not_committed(self):
        """A blocked request must not count toward the window."""
        limiter = RateLimiter()
        policy = _policy(max_requests_per_minute=2)
        limiter.check("u1", tokens=1, policy=policy)
        limiter.check("u1", tokens=1, policy=policy)
        # This is the violation
        limiter.check("u1", tokens=1, policy=policy)
        # Usage should still be 2 (not 3), since the 3rd was rejected
        usage = limiter.get_usage("u1")
        assert usage["requests_last_minute"] == 2

    def test_window_resets_after_expiry(self):
        limiter = RateLimiter()
        policy = _policy(max_requests_per_minute=2)
        now = time.monotonic()
        # Simulate two requests 70 seconds ago (outside 1-min window)
        with patch.object(limiter, "_req_ts") as mock_req_ts:
            import collections
            old_ts = now - 70
            mock_req_ts.__getitem__ = lambda self, key: collections.deque([old_ts, old_ts])
            mock_req_ts.__setitem__ = lambda self, k, v: None
        # Start fresh limiter, manually seed old timestamps
        limiter2 = RateLimiter()
        from collections import deque
        limiter2._req_ts["u1"] = deque([now - 70, now - 70])
        # Both old entries are outside 60s window — should pass
        assert limiter2.check("u1", tokens=1, policy=policy) is None


# ── Token rate limit ──────────────────────────────────────────────────────────

class TestTokenRateLimit:
    def test_within_token_limit_passes(self):
        limiter = RateLimiter()
        policy = _policy(max_tokens_per_hour=100)
        assert limiter.check("u1", tokens=50, policy=policy) is None
        assert limiter.check("u1", tokens=50, policy=policy) is None

    def test_exceeding_token_limit_returns_violation(self):
        limiter = RateLimiter()
        policy = _policy(max_tokens_per_hour=100)
        limiter.check("u1", tokens=100, policy=policy)
        v = limiter.check("u1", tokens=1, policy=policy)
        assert v is not None
        assert v.rule_name == "rate_limit.tokens_per_hour"

    def test_token_violation_message_contains_user_id(self):
        limiter = RateLimiter()
        policy = _policy(max_tokens_per_hour=10)
        limiter.check("alice123", tokens=10, policy=policy)
        v = limiter.check("alice123", tokens=1, policy=policy)
        assert "alice123" in v.message

    def test_token_violation_not_committed(self):
        limiter = RateLimiter()
        policy = _policy(max_tokens_per_hour=100)
        limiter.check("u1", tokens=100, policy=policy)
        # This should be rejected — tokens_last_hour should stay at 100
        limiter.check("u1", tokens=50, policy=policy)
        usage = limiter.get_usage("u1")
        assert usage["tokens_last_hour"] == 100


# ── get_usage ─────────────────────────────────────────────────────────────────

class TestGetUsage:
    def test_usage_reflects_committed_requests(self):
        limiter = RateLimiter()
        policy = _policy()
        limiter.check("u1", tokens=10, policy=policy)
        limiter.check("u1", tokens=20, policy=policy)
        usage = limiter.get_usage("u1")
        assert usage["requests_last_minute"] == 2
        assert usage["tokens_last_hour"] == 30

    def test_usage_for_unknown_user_is_zero(self):
        limiter = RateLimiter()
        usage = limiter.get_usage("nobody")
        assert usage["requests_last_minute"] == 0
        assert usage["tokens_last_hour"] == 0


# ── reset ─────────────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_clears_counters(self):
        limiter = RateLimiter()
        policy = _policy(max_requests_per_minute=2)
        limiter.check("u1", tokens=1, policy=policy)
        limiter.check("u1", tokens=1, policy=policy)
        limiter.reset("u1")
        # After reset, user can make requests again
        assert limiter.check("u1", tokens=1, policy=policy) is None

    def test_reset_unknown_user_is_safe(self):
        limiter = RateLimiter()
        limiter.reset("ghost")  # must not raise


# ── Integration with core ─────────────────────────────────────────────────────

class TestRateLimitInCore:
    async def test_rate_limit_blocks_via_ainvoke(self):
        from unittest.mock import AsyncMock, MagicMock
        from langchain_core.messages import AIMessage, HumanMessage
        from sentinel import Sentinel, SentinelViolation
        from sentinel.policy import (
            AuditPolicy, BudgetPolicy, CircuitBreakerPolicy,
            HallucinationPolicy, InputPolicy, OutputPolicy, SentinelPolicy,
        )

        g = MagicMock()
        g.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content="ok")]})

        policy = SentinelPolicy(
            input=InputPolicy(max_tokens=4096),
            budget=BudgetPolicy(max_cost_usd=10.0, max_tokens_per_run=100_000),
            circuit_breaker=CircuitBreakerPolicy(max_node_repeats=10, max_retries=10),
            rate_limit=RateLimitPolicy(enabled=True, max_requests_per_minute=2),
            output=OutputPolicy(toxicity_check=False),
            hallucination=HallucinationPolicy(enabled=False),
            audit=AuditPolicy(log_all=True, tracely_endpoint=None),
        )
        agent = Sentinel(g, policy=policy)

        # First two calls succeed
        await agent.ainvoke({"messages": [HumanMessage(content="Q")]}, user_id="user1")
        await agent.ainvoke({"messages": [HumanMessage(content="Q")]}, user_id="user1")

        # Third call should be blocked
        with pytest.raises(SentinelViolation) as exc_info:
            await agent.ainvoke({"messages": [HumanMessage(content="Q")]}, user_id="user1")
        assert "rate_limit" in exc_info.value.rule_name

    async def test_different_user_ids_independent(self):
        from unittest.mock import AsyncMock, MagicMock
        from langchain_core.messages import AIMessage, HumanMessage
        from sentinel import Sentinel
        from sentinel.policy import (
            AuditPolicy, BudgetPolicy, CircuitBreakerPolicy,
            HallucinationPolicy, InputPolicy, OutputPolicy, SentinelPolicy,
        )

        g = MagicMock()
        g.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content="ok")]})

        policy = SentinelPolicy(
            input=InputPolicy(max_tokens=4096),
            budget=BudgetPolicy(max_cost_usd=10.0, max_tokens_per_run=100_000),
            circuit_breaker=CircuitBreakerPolicy(max_node_repeats=10, max_retries=10),
            rate_limit=RateLimitPolicy(enabled=True, max_requests_per_minute=1),
            output=OutputPolicy(toxicity_check=False),
            hallucination=HallucinationPolicy(enabled=False),
            audit=AuditPolicy(log_all=True, tracely_endpoint=None),
        )
        agent = Sentinel(g, policy=policy)

        await agent.ainvoke({"messages": [HumanMessage(content="Q")]}, user_id="alice")
        # Bob is a different user — should not be affected by Alice's quota
        result = await agent.ainvoke({"messages": [HumanMessage(content="Q")]}, user_id="bob")
        assert "_sentinel_run_id" in result
