from __future__ import annotations

from collections import defaultdict

from sentinel.stores.base import ViolationStore
from sentinel.violation import ViolationLog


class MemoryViolationStore(ViolationStore):
    """In-process violation store backed by a plain dict.

    Useful for tests and as a named handle when you want store-API compatibility
    without touching disk or a network service.
    """

    def __init__(self) -> None:
        self._data: dict[str, list[ViolationLog]] = defaultdict(list)

    async def save(self, run_id: str, log: ViolationLog) -> None:
        self._data[run_id].append(log)

    async def get(
        self, run_id: str, *, include_shadow: bool = True
    ) -> list[ViolationLog]:
        logs = list(self._data.get(run_id, []))
        if include_shadow:
            return logs
        return [v for v in logs if not v.shadow]
