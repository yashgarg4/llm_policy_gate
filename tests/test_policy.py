"""Phase 3 tests — policy loading, validation, and hot-reload."""
from __future__ import annotations

import textwrap
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sentinel.policy import (
    AuditPolicy,
    BudgetPolicy,
    CircuitBreakerPolicy,
    HallucinationPolicy,
    InputPolicy,
    OutputPolicy,
    SentinelPolicy,
    load_policy,
)
from sentinel.violation import ViolationAction
from sentinel.watcher import PolicyWatcher


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_policy(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "policy.yaml"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


# ── Valid policy loading ──────────────────────────────────────────────────────

class TestPolicyLoading:
    def test_full_policy_loads_correctly(self, tmp_path):
        p = _write_policy(tmp_path, """
            service: my-agent
            input:
              max_tokens: 1024
              block_patterns:
                - "ignore previous instructions"
              pii_detection: true
              pii_action: REDACT
            budget:
              max_cost_usd: 2.50
              max_tokens_per_run: 50000
              action: BLOCK
            circuit_breaker:
              max_node_repeats: 4
              max_retries: 2
              action: ABORT
            output:
              toxicity_check: true
              toxicity_action: FLAG
              topic_guardrail:
                - technology
                - science
            hallucination:
              enabled: true
              threshold: 0.8
              action: FLAG
            audit:
              log_all: true
              tracely_endpoint: "http://localhost:4318"
        """)
        policy = load_policy(p)
        assert policy.service == "my-agent"
        assert policy.input.max_tokens == 1024
        assert policy.input.pii_detection is True
        assert policy.input.pii_action == ViolationAction.REDACT
        assert "ignore previous instructions" in policy.input.block_patterns
        assert policy.budget.max_cost_usd == 2.50
        assert policy.budget.max_tokens_per_run == 50_000
        assert policy.circuit_breaker.max_node_repeats == 4
        assert policy.circuit_breaker.max_retries == 2
        assert policy.output.toxicity_check is True
        assert policy.output.topic_guardrail == ["technology", "science"]
        assert policy.hallucination.enabled is True
        assert policy.hallucination.threshold == 0.8
        assert policy.audit.tracely_endpoint == "http://localhost:4318"

    def test_minimal_policy_uses_defaults(self, tmp_path):
        p = _write_policy(tmp_path, "service: minimal")
        policy = load_policy(p)
        assert policy.service == "minimal"
        assert policy.input.max_tokens == 4096
        assert policy.input.pii_detection is False
        assert policy.budget.max_cost_usd == 1.0
        assert policy.circuit_breaker.max_node_repeats == 5
        assert policy.hallucination.enabled is False
        assert policy.audit.log_all is True

    def test_empty_yaml_uses_all_defaults(self, tmp_path):
        p = tmp_path / "policy.yaml"
        p.write_text("", encoding="utf-8")
        policy = load_policy(p)
        assert isinstance(policy, SentinelPolicy)
        assert policy.service == "sentinel"

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Policy file not found"):
            load_policy(tmp_path / "nonexistent.yaml")

    def test_pydantic_models_all_instantiate_with_defaults(self):
        assert InputPolicy()
        assert BudgetPolicy()
        assert CircuitBreakerPolicy()
        assert OutputPolicy()
        assert HallucinationPolicy()
        assert AuditPolicy()
        assert SentinelPolicy()


# ── Validation errors ─────────────────────────────────────────────────────────

class TestPolicyValidation:
    def test_invalid_action_value_raises_clear_error(self, tmp_path):
        p = _write_policy(tmp_path, """
            input:
              pii_action: EXPLODE
        """)
        with pytest.raises(ValueError) as exc_info:
            load_policy(p)
        msg = str(exc_info.value)
        assert "Invalid policy file" in msg

    def test_negative_max_tokens_raises_error(self, tmp_path):
        p = _write_policy(tmp_path, """
            input:
              max_tokens: -5
        """)
        with pytest.raises(ValueError) as exc_info:
            load_policy(p)
        assert "Invalid policy file" in str(exc_info.value)

    def test_zero_max_tokens_raises_error(self, tmp_path):
        p = _write_policy(tmp_path, """
            input:
              max_tokens: 0
        """)
        with pytest.raises(ValueError):
            load_policy(p)

    def test_threshold_above_1_raises_error(self):
        with pytest.raises(Exception):
            HallucinationPolicy(threshold=1.5)

    def test_threshold_below_0_raises_error(self):
        with pytest.raises(Exception):
            HallucinationPolicy(threshold=-0.1)

    def test_threshold_boundary_values_accepted(self):
        assert HallucinationPolicy(threshold=0.0).threshold == 0.0
        assert HallucinationPolicy(threshold=1.0).threshold == 1.0

    def test_invalid_yaml_syntax_raises_value_error(self, tmp_path):
        p = tmp_path / "policy.yaml"
        p.write_text("service: [unclosed", encoding="utf-8")
        with pytest.raises(ValueError, match="YAML parse error"):
            load_policy(p)

    def test_yaml_non_mapping_raises_value_error(self, tmp_path):
        p = tmp_path / "policy.yaml"
        p.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must be a YAML mapping"):
            load_policy(p)

    def test_negative_budget_raises_error(self, tmp_path):
        p = _write_policy(tmp_path, """
            budget:
              max_cost_usd: -1.0
        """)
        with pytest.raises(ValueError):
            load_policy(p)

    def test_error_message_includes_field_path(self, tmp_path):
        p = _write_policy(tmp_path, """
            input:
              max_tokens: -1
        """)
        with pytest.raises(ValueError) as exc_info:
            load_policy(p)
        # Should mention the problematic field
        assert "input" in str(exc_info.value) or "max_tokens" in str(exc_info.value)


# ── Hot-reload ────────────────────────────────────────────────────────────────

class TestPolicyHotReload:
    def test_watcher_calls_callback_on_file_change(self, tmp_path):
        p = _write_policy(tmp_path, "service: original\nbudget:\n  max_cost_usd: 1.0\n")

        received: list[SentinelPolicy] = []
        watcher = PolicyWatcher(p, on_reload=received.append, debounce_seconds=0.05)
        watcher.start()
        try:
            time.sleep(0.1)
            p.write_text("service: updated\nbudget:\n  max_cost_usd: 9.99\n", encoding="utf-8")
            # Give watchdog + debounce time to fire
            deadline = time.monotonic() + 3.0
            while not received and time.monotonic() < deadline:
                time.sleep(0.1)
        finally:
            watcher.stop()

        assert len(received) >= 1
        assert received[-1].service == "updated"
        assert received[-1].budget.max_cost_usd == 9.99

    def test_watcher_debounces_rapid_changes(self, tmp_path):
        p = _write_policy(tmp_path, "service: v0\n")

        call_times: list[float] = []

        def on_reload(_: SentinelPolicy) -> None:
            call_times.append(time.monotonic())

        watcher = PolicyWatcher(p, on_reload=on_reload, debounce_seconds=0.3)
        watcher.start()
        try:
            time.sleep(0.1)
            # Write 5 times rapidly — debounce should collapse them
            for i in range(5):
                p.write_text(f"service: v{i}\n", encoding="utf-8")
                time.sleep(0.02)
            time.sleep(0.6)
        finally:
            watcher.stop()

        # Debounce should result in far fewer callbacks than writes
        assert len(call_times) <= 3

    def test_sentinel_policy_updates_atomically(self, tmp_path):
        """Sentinel.policy property reflects reload without restart."""
        p = _write_policy(tmp_path, "service: before\n")

        from sentinel.core import Sentinel

        mock_graph = MagicMock()
        agent = Sentinel(mock_graph, policy=p)
        try:
            assert agent.policy.service == "before"

            p.write_text("service: after\nbudget:\n  max_cost_usd: 42.0\n", encoding="utf-8")
            deadline = time.monotonic() + 3.0
            while agent.policy.service != "after" and time.monotonic() < deadline:
                time.sleep(0.1)

            assert agent.policy.service == "after"
            assert agent.policy.budget.max_cost_usd == 42.0
        finally:
            agent.stop_watcher()

    def test_watcher_invalid_reload_does_not_crash(self, tmp_path):
        """A bad YAML file on reload logs an error but doesn't crash the watcher."""
        p = _write_policy(tmp_path, "service: good\n")

        received: list[SentinelPolicy] = []
        watcher = PolicyWatcher(p, on_reload=received.append, debounce_seconds=0.05)
        watcher.start()
        try:
            time.sleep(0.1)
            p.write_text("service: [invalid yaml", encoding="utf-8")
            time.sleep(0.4)
            # No callback should have fired for the bad file
            assert len(received) == 0
            # Watcher should still be alive
            assert watcher._observer.is_alive()
        finally:
            watcher.stop()
