from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentinel.violation import ViolationLog


class ViolationStore(ABC):
    """Async persistence layer for ViolationLog entries."""

    @abstractmethod
    async def save(self, run_id: str, log: "ViolationLog") -> None:
        """Persist a single violation."""

    @abstractmethod
    async def get(
        self, run_id: str, *, include_shadow: bool = True
    ) -> list["ViolationLog"]:
        """Return all violations for run_id, optionally filtering out shadow entries."""

    async def close(self) -> None:
        """Release any underlying connections. Default is a no-op."""
