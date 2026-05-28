"""SQLite-backed ViolationStore using aiosqlite.

Install: pip install aiosqlite  (or sentinel-ai[sqlite])
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from sentinel.stores.base import ViolationStore
from sentinel.violation import ViolationAction, ViolationLog, ViolationSeverity

_DDL = """
CREATE TABLE IF NOT EXISTS sentinel_violations (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT    NOT NULL,
    rule_name         TEXT    NOT NULL,
    action            TEXT    NOT NULL,
    severity          TEXT    NOT NULL,
    message           TEXT    NOT NULL,
    offending_content TEXT    NOT NULL DEFAULT '',
    timestamp         TEXT    NOT NULL,
    node_name         TEXT    NOT NULL DEFAULT '',
    shadow            INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_sv_run_id ON sentinel_violations(run_id);
"""


class SQLiteViolationStore(ViolationStore):
    """Persists violations to a local SQLite database via aiosqlite.

    Pass db_path=':memory:' for an in-process ephemeral store (useful in tests).
    """

    def __init__(self, db_path: str = "sentinel_violations.db") -> None:
        self._db_path = db_path
        self._conn: Any = None  # aiosqlite.Connection

    async def _connect(self) -> None:
        if self._conn is None:
            import aiosqlite

            self._conn = await aiosqlite.connect(self._db_path)
            await self._conn.executescript(_DDL)
            await self._conn.commit()

    async def save(self, run_id: str, log: ViolationLog) -> None:
        await self._connect()
        await self._conn.execute(
            """
            INSERT INTO sentinel_violations
                (run_id, rule_name, action, severity, message,
                 offending_content, timestamp, node_name, shadow)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                log.rule_name,
                log.action.value,
                log.severity.value,
                log.message,
                log.offending_content,
                log.timestamp.isoformat(),
                log.node_name,
                int(log.shadow),
            ),
        )
        await self._conn.commit()

    async def get(
        self, run_id: str, *, include_shadow: bool = True
    ) -> list[ViolationLog]:
        await self._connect()
        if include_shadow:
            sql = "SELECT * FROM sentinel_violations WHERE run_id = ? ORDER BY id"
            params: tuple = (run_id,)
        else:
            sql = "SELECT * FROM sentinel_violations WHERE run_id = ? AND shadow = 0 ORDER BY id"
            params = (run_id,)
        cursor = await self._conn.execute(sql, params)
        rows = await cursor.fetchall()
        return [_row_to_log(row) for row in rows]

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None


def _row_to_log(row: tuple) -> ViolationLog:
    _, run_id, rule_name, action, severity, message, offending_content, timestamp, node_name, shadow = row
    return ViolationLog(
        run_id=run_id,
        rule_name=rule_name,
        action=ViolationAction(action),
        severity=ViolationSeverity(severity),
        message=message,
        offending_content=offending_content,
        timestamp=datetime.fromisoformat(timestamp),
        node_name=node_name,
        shadow=bool(shadow),
    )
