from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator

from sentinel.violation import ViolationAction


class InputPolicy(BaseModel):
    max_tokens: int = Field(default=4096, gt=0)
    block_patterns: list[str] = Field(default_factory=list)
    pii_detection: bool = False
    pii_action: ViolationAction = ViolationAction.REDACT
    # Tier 2b: semantic injection detection
    semantic_injection: bool = False
    semantic_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    semantic_action: ViolationAction = ViolationAction.BLOCK


class BudgetPolicy(BaseModel):
    max_cost_usd: float = Field(default=1.0, gt=0)
    max_tokens_per_run: int = Field(default=100_000, gt=0)
    action: ViolationAction = ViolationAction.BLOCK


class CircuitBreakerPolicy(BaseModel):
    max_node_repeats: int = Field(default=5, gt=0)
    max_retries: int = Field(default=3, gt=0)
    action: ViolationAction = ViolationAction.ABORT


class RateLimitPolicy(BaseModel):
    """Cross-run sliding-window limits keyed by user_id."""
    enabled: bool = False
    max_requests_per_minute: int = Field(default=60, gt=0)
    max_tokens_per_hour: int = Field(default=500_000, gt=0)
    action: ViolationAction = ViolationAction.BLOCK


class OutputPolicy(BaseModel):
    schema_validation: bool = False
    toxicity_check: bool = True
    toxicity_action: ViolationAction = ViolationAction.FLAG
    topic_guardrail: Optional[list[str]] = None
    topic_action: ViolationAction = ViolationAction.FLAG
    # Tier 2c: JSON Schema string to validate structured LLM output against
    output_schema: Optional[str] = None
    schema_action: ViolationAction = ViolationAction.FLAG


class HallucinationPolicy(BaseModel):
    enabled: bool = False
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    action: ViolationAction = ViolationAction.FLAG

    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        return v


class AuditPolicy(BaseModel):
    log_all: bool = True
    tracely_endpoint: Optional[str] = None


class SentinelPolicy(BaseModel):
    service: str = "sentinel"
    input: InputPolicy = Field(default_factory=InputPolicy)
    budget: BudgetPolicy = Field(default_factory=BudgetPolicy)
    circuit_breaker: CircuitBreakerPolicy = Field(default_factory=CircuitBreakerPolicy)
    rate_limit: RateLimitPolicy = Field(default_factory=RateLimitPolicy)
    output: OutputPolicy = Field(default_factory=OutputPolicy)
    hallucination: HallucinationPolicy = Field(default_factory=HallucinationPolicy)
    audit: AuditPolicy = Field(default_factory=AuditPolicy)


def load_policy(path: str | Path) -> SentinelPolicy:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Policy file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Cannot read {path}: file is not UTF-8 encoded. "
            f"Re-save it as UTF-8 and try again. ({exc})"
        ) from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML parse error in {path}: {exc}") from exc

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"Policy file {path} must be a YAML mapping, got {type(raw).__name__}")

    try:
        return SentinelPolicy.model_validate(raw)
    except Exception as exc:
        # Re-raise with field-level detail from Pydantic
        from pydantic import ValidationError

        if isinstance(exc, ValidationError):
            lines = [f"Invalid policy file {path}:"]
            for err in exc.errors():
                loc = " -> ".join(str(p) for p in err["loc"]) if err["loc"] else "root"
                lines.append(f"  [{loc}] {err['msg']} (got {err.get('input')!r})")
            raise ValueError("\n".join(lines)) from exc
        raise ValueError(f"Invalid policy file {path}: {exc}") from exc
