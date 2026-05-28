from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace as _dc_replace
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class ViolationAction(str, Enum):
    BLOCK = "BLOCK"
    FLAG = "FLAG"
    REDACT = "REDACT"
    WARN = "WARN"
    ABORT = "ABORT"


class ViolationSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class SentinelViolation(Exception):
    def __init__(
        self,
        rule_name: str,
        action: ViolationAction,
        severity: ViolationSeverity,
        message: str,
        offending_content: str = "",
        timestamp: Optional[datetime] = None,
        run_id: str = "",
    ) -> None:
        self.rule_name = rule_name
        self.action = action
        self.severity = severity
        self.message = message
        self.offending_content = offending_content
        self.timestamp = timestamp or datetime.now(timezone.utc)
        self.run_id = run_id
        super().__init__(message)

    def __repr__(self) -> str:
        return (
            f"SentinelViolation(rule={self.rule_name!r}, action={self.action.value}, "
            f"severity={self.severity.value}, message={self.message!r})"
        )


@dataclass
class ViolationLog:
    rule_name: str
    action: ViolationAction
    severity: ViolationSeverity
    message: str
    offending_content: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    node_name: str = ""
    shadow: bool = False  # True when observed under shadow policy — never enforced

    def as_shadow(self) -> "ViolationLog":
        """Return a copy of this log marked as a shadow violation."""
        return _dc_replace(self, shadow=True)

    def to_sentinel_violation(self) -> SentinelViolation:
        return SentinelViolation(
            rule_name=self.rule_name,
            action=self.action,
            severity=self.severity,
            message=self.message,
            offending_content=self.offending_content,
            timestamp=self.timestamp,
            run_id=self.run_id,
        )

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "rule_name": self.rule_name,
            "action": self.action.value,
            "severity": self.severity.value,
            "message": self.message,
            "offending_content": self.offending_content,
            "timestamp": self.timestamp.isoformat(),
            "node_name": self.node_name,
            "shadow": self.shadow,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ViolationLog":
        return cls(
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
