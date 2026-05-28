"""Tests for sentinel.metrics (Tier 2d) — Prometheus counters and gauges."""
from __future__ import annotations

import pytest

from sentinel.metrics import (
    is_available,
    record_violation,
    update_budget,
    update_rate_usage,
    start_metrics_server,
)
from sentinel.violation import ViolationAction, ViolationLog, ViolationSeverity


def _violation(**kw) -> ViolationLog:
    defaults = dict(
        rule_name="test.rule",
        action=ViolationAction.FLAG,
        severity=ViolationSeverity.MEDIUM,
        message="Test",
        offending_content="bad",
        run_id="run-1",
        node_name="input",
    )
    defaults.update(kw)
    return ViolationLog(**defaults)


# ── Availability ──────────────────────────────────────────────────────────────

class TestAvailability:
    def test_is_available_returns_bool(self):
        assert isinstance(is_available(), bool)

    def test_prometheus_is_installed(self):
        # prometheus_client is listed as optional — verify it installed
        assert is_available() is True


# ── record_violation ──────────────────────────────────────────────────────────

class TestRecordViolation:
    def test_does_not_raise(self):
        try:
            record_violation(_violation(), service="sentinel-test")
        except Exception as exc:
            pytest.fail(f"record_violation raised: {exc}")

    def test_counter_incremented(self):
        if not is_available():
            pytest.skip("prometheus_client not available")
        from sentinel.metrics import _violations_total
        v = _violation(rule_name="metrics.test.rule")
        before = _violations_total.labels(
            rule_name="metrics.test.rule",
            action=ViolationAction.FLAG.value,
            severity=ViolationSeverity.MEDIUM.value,
            service="svc-a",
        )._value.get()
        record_violation(v, service="svc-a")
        after = _violations_total.labels(
            rule_name="metrics.test.rule",
            action=ViolationAction.FLAG.value,
            severity=ViolationSeverity.MEDIUM.value,
            service="svc-a",
        )._value.get()
        assert after == before + 1

    def test_all_violation_actions_handled(self):
        for action in ViolationAction:
            try:
                record_violation(_violation(action=action), service="svc")
            except Exception as exc:
                pytest.fail(f"record_violation raised for action={action}: {exc}")

    def test_all_severities_handled(self):
        for sev in ViolationSeverity:
            try:
                record_violation(_violation(severity=sev), service="svc")
            except Exception as exc:
                pytest.fail(f"record_violation raised for severity={sev}: {exc}")

    def test_custom_service_name_used(self):
        if not is_available():
            pytest.skip("prometheus_client not available")
        from sentinel.metrics import _violations_total
        v = _violation(rule_name="metrics.svc.rule")
        record_violation(v, service="my-custom-service")
        # No exception means the label was accepted


# ── update_budget ─────────────────────────────────────────────────────────────

class TestUpdateBudget:
    def test_does_not_raise(self):
        try:
            update_budget("run-99", tokens=500, cost_usd=0.05, service="svc")
        except Exception as exc:
            pytest.fail(f"update_budget raised: {exc}")

    def test_gauge_value_set(self):
        if not is_available():
            pytest.skip("prometheus_client not available")
        from sentinel.metrics import _budget_tokens, _budget_cost_usd
        update_budget("run-gauge-test", tokens=1234, cost_usd=0.123, service="svc-b")
        assert _budget_tokens.labels(run_id="run-gauge-test", service="svc-b")._value.get() == 1234
        assert abs(_budget_cost_usd.labels(run_id="run-gauge-test", service="svc-b")._value.get() - 0.123) < 1e-9


# ── update_rate_usage ─────────────────────────────────────────────────────────

class TestUpdateRateUsage:
    def test_does_not_raise(self):
        try:
            update_rate_usage("user-1", requests_last_minute=5, tokens_last_hour=500)
        except Exception as exc:
            pytest.fail(f"update_rate_usage raised: {exc}")

    def test_gauge_values_set(self):
        if not is_available():
            pytest.skip("prometheus_client not available")
        from sentinel.metrics import _rate_requests, _rate_tokens
        update_rate_usage("uid-x", requests_last_minute=7, tokens_last_hour=888, service="svc-c")
        assert _rate_requests.labels(user_id="uid-x", service="svc-c")._value.get() == 7
        assert _rate_tokens.labels(user_id="uid-x", service="svc-c")._value.get() == 888


# ── start_metrics_server ──────────────────────────────────────────────────────

class TestStartMetricsServer:
    def test_does_not_raise_on_port_conflict(self):
        # Port 0 is special but may vary by OS — just verify it doesn't crash
        try:
            start_metrics_server(port=19999)
        except Exception as exc:
            pytest.fail(f"start_metrics_server raised: {exc}")


# ── Wired into core._record_violation ────────────────────────────────────────

class TestMetricsWiredInCore:
    async def test_violation_increments_counter(self):
        if not is_available():
            pytest.skip("prometheus_client not available")
        from unittest.mock import AsyncMock, MagicMock
        from langchain_core.messages import AIMessage, HumanMessage
        from sentinel import Sentinel, SentinelViolation
        from sentinel.policy import (
            AuditPolicy, BudgetPolicy, CircuitBreakerPolicy,
            HallucinationPolicy, InputPolicy, OutputPolicy,
            RateLimitPolicy, SentinelPolicy,
        )
        from sentinel.metrics import _violations_total

        g = MagicMock()
        g.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content="ok")]})

        policy = SentinelPolicy(
            input=InputPolicy(
                block_patterns=["metrics test pattern"],
                max_tokens=4096,
            ),
            budget=BudgetPolicy(max_cost_usd=10.0, max_tokens_per_run=100_000),
            circuit_breaker=CircuitBreakerPolicy(max_node_repeats=10, max_retries=10),
            rate_limit=RateLimitPolicy(enabled=False),
            output=OutputPolicy(toxicity_check=False),
            hallucination=HallucinationPolicy(enabled=False),
            audit=AuditPolicy(log_all=True, tracely_endpoint=None),
        )
        agent = Sentinel(g, policy=policy)

        before = _violations_total.labels(
            rule_name="input.block_pattern",
            action="BLOCK",
            severity="CRITICAL",
            service="sentinel",
        )._value.get()

        with pytest.raises(SentinelViolation):
            await agent.ainvoke(
                {"messages": [HumanMessage(content="metrics test pattern here")]}
            )

        after = _violations_total.labels(
            rule_name="input.block_pattern",
            action="BLOCK",
            severity="CRITICAL",
            service="sentinel",
        )._value.get()
        assert after == before + 1
