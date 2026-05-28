"""Redis-backed ViolationStore using redis.asyncio.

Install: pip install redis  (or sentinel-ai[redis])

Each run's violations are stored as a Redis list at key
``sentinel:violations:{run_id}`` with a configurable TTL (default 24 h).
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from sentinel.stores.base import ViolationStore
from sentinel.violation import ViolationAction, ViolationLog, ViolationSeverity

_DEFAULT_TTL = 86_400  # 24 hours


class RedisViolationStore(ViolationStore):
    """Persists violations to Redis as JSON-encoded list entries.

    Args:
        redis_url: Connection URL, e.g. ``"redis://localhost:6379"``.
        ttl: Seconds before a run's violation list expires (default 86400).
        key_prefix: Prefix for all Redis keys (default ``"sentinel:violations"``).
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        ttl: int = _DEFAULT_TTL,
        key_prefix: str = "sentinel:violations",
    ) -> None:
        self._redis_url = redis_url
        self._ttl = ttl
        self._key_prefix = key_prefix
        self._client: Any = None  # redis.asyncio.Redis

    def _key(self, run_id: str) -> str:
        return f"{self._key_prefix}:{run_id}"

    async def _connect(self) -> None:
        if self._client is None:
            from redis.asyncio import from_url  # type: ignore[import]

            self._client = await from_url(self._redis_url, decode_responses=True)

    async def save(self, run_id: str, log: ViolationLog) -> None:
        await self._connect()
        key = self._key(run_id)
        await self._client.rpush(key, json.dumps(_log_to_dict(log)))
        await self._client.expire(key, self._ttl)

    async def get(
        self, run_id: str, *, include_shadow: bool = True
    ) -> list[ViolationLog]:
        await self._connect()
        raw: list[str] = await self._client.lrange(self._key(run_id), 0, -1)
        logs = [_dict_to_log(json.loads(r)) for r in raw]
        if include_shadow:
            return logs
        return [v for v in logs if not v.shadow]

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def _log_to_dict(log: ViolationLog) -> dict:
    return {
        "run_id": log.run_id,
        "rule_name": log.rule_name,
        "action": log.action.value,
        "severity": log.severity.value,
        "message": log.message,
        "offending_content": log.offending_content,
        "timestamp": log.timestamp.isoformat(),
        "node_name": log.node_name,
        "shadow": log.shadow,
    }


def _dict_to_log(d: dict) -> ViolationLog:
    return ViolationLog(
        run_id=d["run_id"],
        rule_name=d["rule_name"],
        action=ViolationAction(d["action"]),
        severity=ViolationSeverity(d["severity"]),
        message=d["message"],
        offending_content=d.get("offending_content", ""),
        timestamp=datetime.fromisoformat(d["timestamp"]),
        node_name=d.get("node_name", ""),
        shadow=bool(d.get("shadow", False)),
    )
