"""Tests for telemetry emission — blocking vs daemon, attributes, error tolerance."""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, call

import pytest

from sentinel.telemetry import _emit, emit_violation, shutdown_all
from sentinel.violation import ViolationAction, ViolationLog, ViolationSeverity


def _violation(**kw) -> ViolationLog:
    defaults = dict(
        rule_name="test.rule",
        action=ViolationAction.FLAG,
        severity=ViolationSeverity.MEDIUM,
        message="Test violation",
        offending_content="bad content",
        run_id="run-abc",
        node_name="input",
    )
    defaults.update(kw)
    return ViolationLog(**defaults)


# ── emit_violation dispatch ───────────────────────────────────────────────────

class TestEmitViolationDispatch:
    def test_blocking_true_calls_emit_synchronously(self):
        v = _violation(action=ViolationAction.BLOCK)
        with patch("sentinel.telemetry._emit") as mock_emit:
            emit_violation(v, "http://localhost:8001/v1/traces", blocking=True)
        mock_emit.assert_called_once_with(v, "http://localhost:8001/v1/traces", "sentinel")

    def test_blocking_false_spawns_daemon_thread(self):
        v = _violation()
        with patch("sentinel.telemetry.threading.Thread") as mock_thread_cls:
            mock_thread = MagicMock()
            mock_thread_cls.return_value = mock_thread
            emit_violation(v, "http://localhost:8001/v1/traces", blocking=False)
        mock_thread_cls.assert_called_once()
        assert mock_thread_cls.call_args.kwargs.get("daemon") is True
        mock_thread.start.assert_called_once()

    def test_custom_service_name_forwarded(self):
        v = _violation()
        with patch("sentinel.telemetry._emit") as mock_emit:
            emit_violation(v, "http://example.com/v1/traces", "my-service", blocking=True)
        mock_emit.assert_called_once_with(v, "http://example.com/v1/traces", "my-service")

    def test_default_service_name_is_sentinel(self):
        v = _violation()
        with patch("sentinel.telemetry._emit") as mock_emit:
            emit_violation(v, "http://example.com/v1/traces", blocking=True)
        _, _, svc = mock_emit.call_args.args
        assert svc == "sentinel"

    def test_blocking_false_does_not_block_caller(self):
        """Thread should be started and control returned before _emit completes."""
        started = threading.Event()
        finished = threading.Event()

        def slow_emit(*_):
            started.set()
            finished.wait(timeout=2)

        v = _violation()
        with patch("sentinel.telemetry._emit", side_effect=slow_emit):
            emit_violation(v, "http://example.com/v1/traces", blocking=False)
            # If non-blocking, caller returns before _emit finishes
            assert started.wait(timeout=1)
        finished.set()


# ── _emit error tolerance ─────────────────────────────────────────────────────

class TestEmitErrorTolerance:
    def test_unreachable_endpoint_does_not_raise(self):
        v = _violation()
        # Port 9 is the discard port — guaranteed connection refused
        try:
            _emit(v, "http://localhost:9/v1/traces", "test-service")
        except Exception as exc:
            pytest.fail(f"_emit raised on unreachable endpoint: {exc}")

    def test_empty_offending_content_handled(self):
        v = _violation(offending_content="")
        try:
            _emit(v, "http://localhost:9/v1/traces", "svc")
        except Exception as exc:
            pytest.fail(f"_emit raised on empty offending_content: {exc}")

    def test_long_offending_content_truncated_without_raise(self):
        v = _violation(offending_content="x" * 2000)
        try:
            _emit(v, "http://localhost:9/v1/traces", "svc")
        except Exception as exc:
            pytest.fail(f"_emit raised on long content: {exc}")

    def test_all_violation_actions_handled(self):
        for action in ViolationAction:
            v = _violation(action=action)
            try:
                _emit(v, "http://localhost:9/v1/traces", "svc")
            except Exception as exc:
                pytest.fail(f"_emit raised for action={action}: {exc}")

    def test_all_severities_handled(self):
        for sev in ViolationSeverity:
            v = _violation(severity=sev)
            try:
                _emit(v, "http://localhost:9/v1/traces", "svc")
            except Exception as exc:
                pytest.fail(f"_emit raised for severity={sev}: {exc}")


# ── Shutdown ──────────────────────────────────────────────────────────────────

class TestShutdown:
    def test_shutdown_all_is_idempotent(self):
        shutdown_all()
        shutdown_all()  # second call must not raise

    def test_shutdown_all_clears_providers(self):
        from sentinel.telemetry import _providers
        # Populate a provider
        _emit(_violation(), "http://localhost:9/v1/traces", "test-shutdown-svc")
        shutdown_all()
        assert len(_providers) == 0


# ── Provider caching ──────────────────────────────────────────────────────────

class TestProviderCaching:
    def test_same_endpoint_and_service_reuses_provider(self):
        from sentinel.telemetry import _providers, _get_tracer
        shutdown_all()
        _get_tracer("http://localhost:9/v1/traces", "cache-svc")
        _get_tracer("http://localhost:9/v1/traces", "cache-svc")
        assert len(_providers) == 1

    def test_different_service_names_create_separate_providers(self):
        from sentinel.telemetry import _providers, _get_tracer
        shutdown_all()
        _get_tracer("http://localhost:9/v1/traces", "svc-a")
        _get_tracer("http://localhost:9/v1/traces", "svc-b")
        assert len(_providers) == 2
        shutdown_all()
