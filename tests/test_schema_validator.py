"""Tests for output JSON Schema validation (Tier 2c)."""
from __future__ import annotations

import json

import pytest

from sentinel.async_guards.schema_validator import validate
from sentinel.policy import OutputPolicy
from sentinel.violation import ViolationAction


def _policy(schema: str | None = None, action: ViolationAction = ViolationAction.FLAG) -> OutputPolicy:
    return OutputPolicy(
        toxicity_check=False,
        output_schema=schema,
        schema_action=action,
    )


_SIMPLE_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age":  {"type": "integer"},
    },
    "required": ["name", "age"],
    "additionalProperties": False,
})


# ── No schema configured ──────────────────────────────────────────────────────

class TestNoSchema:
    async def test_no_schema_returns_none(self):
        v = await validate('{"name": "Alice", "age": 30}', _policy(schema=None))
        assert v is None

    async def test_empty_text_with_schema_returns_none(self):
        v = await validate("", _policy(schema=_SIMPLE_SCHEMA))
        assert v is None


# ── Valid output ──────────────────────────────────────────────────────────────

class TestValidOutput:
    async def test_valid_json_matching_schema_returns_none(self):
        text = json.dumps({"name": "Alice", "age": 30})
        v = await validate(text, _policy(schema=_SIMPLE_SCHEMA))
        assert v is None

    async def test_nested_schema_valid(self):
        schema = json.dumps({
            "type": "object",
            "properties": {
                "result": {
                    "type": "array",
                    "items": {"type": "string"},
                }
            },
            "required": ["result"],
        })
        text = json.dumps({"result": ["a", "b", "c"]})
        v = await validate(text, _policy(schema=schema))
        assert v is None


# ── Invalid JSON output ───────────────────────────────────────────────────────

class TestInvalidJson:
    async def test_non_json_output_returns_violation(self):
        v = await validate("This is plain text, not JSON.", _policy(schema=_SIMPLE_SCHEMA))
        assert v is not None
        assert v.rule_name == "output.schema_not_json"

    async def test_action_propagated_on_non_json(self):
        v = await validate("plain text", _policy(schema=_SIMPLE_SCHEMA, action=ViolationAction.FLAG))
        assert v.action == ViolationAction.FLAG

    async def test_truncated_json_flagged(self):
        v = await validate('{"name": "Alice"', _policy(schema=_SIMPLE_SCHEMA))
        assert v is not None
        assert v.rule_name == "output.schema_not_json"


# ── Schema mismatch ───────────────────────────────────────────────────────────

class TestSchemaMismatch:
    async def test_missing_required_field_returns_violation(self):
        text = json.dumps({"name": "Alice"})  # missing "age"
        v = await validate(text, _policy(schema=_SIMPLE_SCHEMA))
        assert v is not None
        assert v.rule_name == "output.schema_mismatch"

    async def test_wrong_type_returns_violation(self):
        text = json.dumps({"name": "Alice", "age": "thirty"})  # age should be int
        v = await validate(text, _policy(schema=_SIMPLE_SCHEMA))
        assert v is not None
        assert v.rule_name == "output.schema_mismatch"

    async def test_additional_property_returns_violation(self):
        text = json.dumps({"name": "Alice", "age": 30, "extra": "field"})
        v = await validate(text, _policy(schema=_SIMPLE_SCHEMA))
        assert v is not None
        assert v.rule_name == "output.schema_mismatch"

    async def test_violation_action_propagated(self):
        text = json.dumps({"name": "Alice"})  # missing age
        v = await validate(text, _policy(schema=_SIMPLE_SCHEMA, action=ViolationAction.FLAG))
        assert v.action == ViolationAction.FLAG

    async def test_violation_message_is_descriptive(self):
        text = json.dumps({"name": "Alice"})
        v = await validate(text, _policy(schema=_SIMPLE_SCHEMA))
        assert v is not None
        assert len(v.message) > 10  # has some useful content


# ── Invalid schema definition ─────────────────────────────────────────────────

class TestInvalidSchemaDefinition:
    async def test_bad_schema_json_returns_warn_violation(self):
        v = await validate('{"name": "x"}', _policy(schema="not-valid-json{"))
        assert v is not None
        assert v.rule_name == "output.schema_invalid_definition"
        assert v.action == ViolationAction.WARN

    async def test_invalid_json_schema_structure_returns_warn(self):
        # Valid JSON but invalid as a JSON Schema
        bad_schema = json.dumps({"type": "invalidtype12345"})
        v = await validate('{"x": 1}', _policy(schema=bad_schema))
        # May return schema_invalid_definition or schema_mismatch depending on jsonschema version
        assert v is not None


# ── run_id and node_name propagation ─────────────────────────────────────────

class TestMetadata:
    async def test_run_id_propagated(self):
        text = json.dumps({"name": "Alice"})  # missing age
        v = await validate(text, _policy(schema=_SIMPLE_SCHEMA), run_id="r99")
        assert v is not None
        assert v.run_id == "r99"

    async def test_node_name_default_is_output(self):
        text = json.dumps({"name": "Alice"})
        v = await validate(text, _policy(schema=_SIMPLE_SCHEMA))
        assert v is not None
        assert v.node_name == "output"


# ── Integration with core ─────────────────────────────────────────────────────

class TestSchemaInCore:
    async def test_schema_violation_logged_after_run(self):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        from langchain_core.messages import AIMessage, HumanMessage
        from sentinel import Sentinel
        from sentinel.policy import (
            AuditPolicy, BudgetPolicy, CircuitBreakerPolicy,
            HallucinationPolicy, InputPolicy, RateLimitPolicy, SentinelPolicy,
        )

        output = json.dumps({"name": "Alice"})  # missing "age" — schema mismatch
        g = MagicMock()
        g.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content=output)]})

        policy = SentinelPolicy(
            input=InputPolicy(max_tokens=4096),
            budget=BudgetPolicy(max_cost_usd=10.0, max_tokens_per_run=100_000),
            circuit_breaker=CircuitBreakerPolicy(max_node_repeats=10, max_retries=10),
            rate_limit=RateLimitPolicy(enabled=False),
            output=OutputPolicy(
                toxicity_check=False,
                output_schema=_SIMPLE_SCHEMA,
                schema_action=ViolationAction.FLAG,
            ),
            hallucination=HallucinationPolicy(enabled=False),
            audit=AuditPolicy(log_all=True, tracely_endpoint=None),
        )
        agent = Sentinel(g, policy=policy)
        result = await agent.ainvoke({"messages": [HumanMessage(content="Summarize")]})
        run_id = result["_sentinel_run_id"]

        await asyncio.sleep(0.3)  # let async guards complete
        violations = await agent.get_violations(run_id)
        assert any(v.rule_name == "output.schema_mismatch" for v in violations)
