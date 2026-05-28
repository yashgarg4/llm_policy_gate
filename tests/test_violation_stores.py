"""Tests for persistent ViolationStore backends (Tier 3c).

Coverage:
  - MemoryViolationStore  — in-process dict-backed store
  - SQLiteViolationStore  — aiosqlite, using ':memory:' so no temp files needed
  - RedisViolationStore   — redis.asyncio client mocked via unittest.mock
  - Sentinel integration  — store= param wires write-through; get_violations()
                            falls back to store for historical run IDs
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

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
from sentinel.stores.memory import MemoryViolationStore
from sentinel.stores.sqlite import SQLiteViolationStore
from sentinel.stores.redis_store import RedisViolationStore
from sentinel.violation import ViolationAction, ViolationLog, ViolationSeverity


# ── Helpers ───────────────────────────────────────────────────────────────────

def _log(
    rule_name: str = "test.rule",
    run_id: str = "run-1",
    shadow: bool = False,
) -> ViolationLog:
    return ViolationLog(
        rule_name=rule_name,
        action=ViolationAction.BLOCK,
        severity=ViolationSeverity.HIGH,
        message="test violation",
        run_id=run_id,
        shadow=shadow,
    )


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


# ── MemoryViolationStore ──────────────────────────────────────────────────────

class TestMemoryViolationStore:
    async def test_save_and_get(self):
        store = MemoryViolationStore()
        log = _log(run_id="r1")
        await store.save("r1", log)
        result = await store.get("r1")
        assert len(result) == 1
        assert result[0].rule_name == "test.rule"

    async def test_unknown_run_returns_empty(self):
        store = MemoryViolationStore()
        result = await store.get("nonexistent")
        assert result == []

    async def test_multiple_violations_per_run(self):
        store = MemoryViolationStore()
        for i in range(3):
            await store.save("r1", _log(rule_name=f"rule.{i}", run_id="r1"))
        result = await store.get("r1")
        assert len(result) == 3

    async def test_runs_are_isolated(self):
        store = MemoryViolationStore()
        await store.save("r1", _log(run_id="r1"))
        await store.save("r2", _log(run_id="r2"))
        assert len(await store.get("r1")) == 1
        assert len(await store.get("r2")) == 1

    async def test_include_shadow_true_returns_all(self):
        store = MemoryViolationStore()
        await store.save("r1", _log(run_id="r1", shadow=False))
        await store.save("r1", _log(run_id="r1", shadow=True))
        result = await store.get("r1", include_shadow=True)
        assert len(result) == 2

    async def test_include_shadow_false_filters_shadow(self):
        store = MemoryViolationStore()
        await store.save("r1", _log(run_id="r1", shadow=False))
        await store.save("r1", _log(run_id="r1", shadow=True))
        result = await store.get("r1", include_shadow=False)
        assert len(result) == 1
        assert not result[0].shadow

    async def test_close_is_noop(self):
        store = MemoryViolationStore()
        await store.close()  # must not raise


# ── SQLiteViolationStore ──────────────────────────────────────────────────────

class TestSQLiteViolationStore:
    async def test_save_and_get(self):
        store = SQLiteViolationStore(":memory:")
        log = _log(run_id="r1")
        await store.save("r1", log)
        result = await store.get("r1")
        assert len(result) == 1
        v = result[0]
        assert v.rule_name == "test.rule"
        assert v.run_id == "r1"
        assert v.action == ViolationAction.BLOCK
        assert v.severity == ViolationSeverity.HIGH
        assert v.shadow is False
        await store.close()

    async def test_unknown_run_returns_empty(self):
        store = SQLiteViolationStore(":memory:")
        result = await store.get("nonexistent")
        assert result == []
        await store.close()

    async def test_multiple_violations_ordered_by_insert(self):
        store = SQLiteViolationStore(":memory:")
        rules = ["rule.a", "rule.b", "rule.c"]
        for r in rules:
            await store.save("r1", _log(rule_name=r, run_id="r1"))
        result = await store.get("r1")
        assert [v.rule_name for v in result] == rules
        await store.close()

    async def test_include_shadow_false_filters(self):
        store = SQLiteViolationStore(":memory:")
        await store.save("r1", _log(run_id="r1", shadow=False))
        await store.save("r1", _log(run_id="r1", shadow=True))
        enforced = await store.get("r1", include_shadow=False)
        assert len(enforced) == 1
        assert not enforced[0].shadow
        await store.close()

    async def test_include_shadow_true_returns_all(self):
        store = SQLiteViolationStore(":memory:")
        await store.save("r1", _log(run_id="r1", shadow=False))
        await store.save("r1", _log(run_id="r1", shadow=True))
        all_v = await store.get("r1", include_shadow=True)
        assert len(all_v) == 2
        await store.close()

    async def test_runs_isolated(self):
        store = SQLiteViolationStore(":memory:")
        await store.save("r1", _log(run_id="r1"))
        await store.save("r2", _log(run_id="r2"))
        assert len(await store.get("r1")) == 1
        assert len(await store.get("r2")) == 1
        await store.close()

    async def test_roundtrip_preserves_all_fields(self):
        store = SQLiteViolationStore(":memory:")
        ts = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        original = ViolationLog(
            rule_name="input.block_pattern",
            action=ViolationAction.ABORT,
            severity=ViolationSeverity.CRITICAL,
            message="blocked",
            offending_content="bad content",
            timestamp=ts,
            run_id="round-trip-1",
            node_name="input",
            shadow=True,
        )
        await store.save("round-trip-1", original)
        result = await store.get("round-trip-1")
        assert len(result) == 1
        v = result[0]
        assert v.rule_name == original.rule_name
        assert v.action == original.action
        assert v.severity == original.severity
        assert v.message == original.message
        assert v.offending_content == original.offending_content
        assert v.timestamp == original.timestamp
        assert v.run_id == original.run_id
        assert v.node_name == original.node_name
        assert v.shadow is True
        await store.close()

    async def test_close_then_reopen_same_path(self, tmp_path):
        db_file = str(tmp_path / "violations.db")
        store = SQLiteViolationStore(db_file)
        await store.save("r1", _log(run_id="r1"))
        await store.close()

        store2 = SQLiteViolationStore(db_file)
        result = await store2.get("r1")
        assert len(result) == 1
        await store2.close()


# ── RedisViolationStore ───────────────────────────────────────────────────────

class TestRedisViolationStore:
    """Redis tests use a fully mocked redis.asyncio client — no real Redis needed."""

    def _make_store_with_mock_client(self) -> tuple[RedisViolationStore, MagicMock]:
        store = RedisViolationStore.__new__(RedisViolationStore)
        store._redis_url = "redis://localhost:6379"
        store._ttl = 86400
        store._key_prefix = "sentinel:violations"
        mock_client = MagicMock()
        mock_client.rpush = AsyncMock(return_value=1)
        mock_client.expire = AsyncMock(return_value=True)
        mock_client.lrange = AsyncMock(return_value=[])
        mock_client.aclose = AsyncMock()
        store._client = mock_client
        return store, mock_client

    async def test_save_calls_rpush_and_expire(self):
        store, client = self._make_store_with_mock_client()
        log = _log(run_id="r1")
        await store.save("r1", log)
        client.rpush.assert_called_once()
        key_arg = client.rpush.call_args[0][0]
        assert key_arg == "sentinel:violations:r1"
        client.expire.assert_called_once_with("sentinel:violations:r1", 86400)

    async def test_get_returns_empty_for_unknown_run(self):
        store, client = self._make_store_with_mock_client()
        client.lrange = AsyncMock(return_value=[])
        result = await store.get("nonexistent")
        assert result == []

    async def test_get_deserializes_stored_entries(self):
        store, client = self._make_store_with_mock_client()
        original = _log(run_id="r1", shadow=False)
        import json
        from sentinel.stores.redis_store import _log_to_dict
        payload = json.dumps(_log_to_dict(original))
        client.lrange = AsyncMock(return_value=[payload])
        result = await store.get("r1")
        assert len(result) == 1
        assert result[0].rule_name == original.rule_name
        assert result[0].run_id == "r1"

    async def test_include_shadow_false_filters(self):
        store, client = self._make_store_with_mock_client()
        import json
        from sentinel.stores.redis_store import _log_to_dict
        entries = [
            json.dumps(_log_to_dict(_log(run_id="r1", shadow=False))),
            json.dumps(_log_to_dict(_log(run_id="r1", shadow=True))),
        ]
        client.lrange = AsyncMock(return_value=entries)
        result = await store.get("r1", include_shadow=False)
        assert len(result) == 1
        assert not result[0].shadow

    async def test_close_calls_aclose(self):
        store, client = self._make_store_with_mock_client()
        await store.close()
        client.aclose.assert_called_once()
        assert store._client is None

    async def test_custom_key_prefix(self):
        store = RedisViolationStore.__new__(RedisViolationStore)
        store._redis_url = "redis://localhost"
        store._ttl = 3600
        store._key_prefix = "myapp:sentinel"
        mock_client = MagicMock()
        mock_client.rpush = AsyncMock(return_value=1)
        mock_client.expire = AsyncMock(return_value=True)
        store._client = mock_client
        await store.save("abc", _log(run_id="abc"))
        key_arg = mock_client.rpush.call_args[0][0]
        assert key_arg == "myapp:sentinel:abc"


# ── ViolationLog serialization ────────────────────────────────────────────────

class TestViolationLogSerialization:
    def test_to_dict_round_trip(self):
        original = ViolationLog(
            rule_name="rate_limit.requests_per_minute",
            action=ViolationAction.FLAG,
            severity=ViolationSeverity.MEDIUM,
            message="rate limited",
            offending_content="user=x",
            run_id="abc-123",
            node_name="input",
            shadow=True,
        )
        restored = ViolationLog.from_dict(original.to_dict())
        assert restored.rule_name == original.rule_name
        assert restored.action == original.action
        assert restored.severity == original.severity
        assert restored.message == original.message
        assert restored.offending_content == original.offending_content
        assert restored.run_id == original.run_id
        assert restored.node_name == original.node_name
        assert restored.shadow == original.shadow
        assert restored.timestamp == original.timestamp

    def test_to_dict_contains_all_keys(self):
        d = _log().to_dict()
        expected_keys = {
            "run_id", "rule_name", "action", "severity", "message",
            "offending_content", "timestamp", "node_name", "shadow",
        }
        assert expected_keys == set(d.keys())


# ── SentinelViolation.run_id ──────────────────────────────────────────────────

class TestSentinelViolationRunId:
    def test_run_id_propagated_from_log(self):
        log = ViolationLog(
            rule_name="input.block_pattern",
            action=ViolationAction.BLOCK,
            severity=ViolationSeverity.CRITICAL,
            message="blocked",
            run_id="my-run-123",
        )
        exc = log.to_sentinel_violation()
        assert exc.run_id == "my-run-123"

    def test_run_id_default_empty(self):
        exc = SentinelViolation(
            rule_name="x", action=ViolationAction.BLOCK,
            severity=ViolationSeverity.LOW, message="m",
        )
        assert exc.run_id == ""


# ── Sentinel integration with store ──────────────────────────────────────────

class TestSentinelWithStore:
    async def test_default_store_is_none(self):
        agent = Sentinel(_graph(), policy=_policy())
        assert agent._store is None

    async def test_custom_store_accepted(self):
        store = MemoryViolationStore()
        agent = Sentinel(_graph(), policy=_policy(), store=store)
        assert agent._store is store

    async def test_violation_written_to_store_on_block(self):
        store = MemoryViolationStore()
        agent = Sentinel(
            _graph(),
            policy=_policy(input=InputPolicy(block_patterns=["bad"])),
            store=store,
        )
        exc = None
        try:
            await agent.ainvoke({"messages": [HumanMessage(content="bad")]})
        except SentinelViolation as e:
            exc = e

        assert exc is not None
        # Let the background write-through task complete
        await asyncio.sleep(0)
        stored = await store.get(exc.run_id)
        assert any(v.rule_name == "input.block_pattern" for v in stored)

    async def test_clean_run_no_store_writes(self):
        store = MemoryViolationStore()
        agent = Sentinel(_graph(), policy=_policy(), store=store)
        result = await agent.ainvoke({"messages": [HumanMessage(content="hello")]})
        run_id = result["_sentinel_run_id"]
        await asyncio.sleep(0)
        stored = await store.get(run_id)
        assert stored == []

    async def test_get_violations_falls_back_to_store(self):
        """get_violations() reads from store when run_id is not in the in-memory dict."""
        store = MemoryViolationStore()
        log = _log(run_id="historical-run")
        await store.save("historical-run", log)

        agent = Sentinel(_graph(), policy=_policy(), store=store)
        # historical-run is NOT in agent._violation_log (different session)
        assert "historical-run" not in agent._violation_log

        result = await agent.get_violations("historical-run")
        assert len(result) == 1
        assert result[0].rule_name == "test.rule"

    async def test_get_violations_prefers_in_memory_over_store(self):
        """In-memory dict takes priority over the store for current-session runs."""
        store = MemoryViolationStore()
        # Seed the store with stale data for this run_id
        stale = _log(rule_name="stale.rule", run_id="current-run")
        await store.save("current-run", stale)

        agent = Sentinel(_graph(), policy=_policy(), store=store)
        # Manually inject a fresh violation into the in-memory dict
        from sentinel.violation import ViolationLog, ViolationSeverity
        fresh = ViolationLog(
            rule_name="fresh.rule",
            action=ViolationAction.FLAG,
            severity=ViolationSeverity.LOW,
            message="fresh",
            run_id="current-run",
        )
        agent._violation_log["current-run"] = [fresh]

        result = await agent.get_violations("current-run")
        assert len(result) == 1
        assert result[0].rule_name == "fresh.rule"

    async def test_sqlite_store_persists_across_sentinel_instances(self, tmp_path):
        db_file = str(tmp_path / "sentinel_test.db")
        store1 = SQLiteViolationStore(db_file)
        agent1 = Sentinel(
            _graph(),
            policy=_policy(input=InputPolicy(block_patterns=["persist-me"])),
            store=store1,
        )
        exc = None
        try:
            await agent1.ainvoke({"messages": [HumanMessage(content="persist-me")]})
        except SentinelViolation as e:
            exc = e
        # Flush background tasks
        await asyncio.gather(*list(agent1._background_tasks))
        await store1.close()

        # Second instance reads from the same SQLite file
        store2 = SQLiteViolationStore(db_file)
        historical = await store2.get(exc.run_id)
        assert any(v.rule_name == "input.block_pattern" for v in historical)
        await store2.close()
