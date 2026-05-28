"""Output JSON Schema validation — Tier 2c.

Validates that the LLM's output is valid JSON that conforms to a JSON Schema
string stored in OutputPolicy.output_schema. Requires the `jsonschema` package.
Falls back gracefully when jsonschema is unavailable (returns None, no crash).
"""
from __future__ import annotations

import json
from typing import Optional

from sentinel.policy import OutputPolicy
from sentinel.violation import ViolationAction, ViolationLog, ViolationSeverity

try:
    import jsonschema
    _HAS_JSONSCHEMA = True
except ImportError:
    jsonschema = None  # type: ignore[assignment]
    _HAS_JSONSCHEMA = False


async def validate(
    text: str,
    policy: OutputPolicy,
    run_id: str = "",
    node_name: str = "output",
) -> Optional[ViolationLog]:
    """Return a ViolationLog if output fails schema validation, else None."""
    if not policy.output_schema or not text:
        return None

    # Step 1: parse schema definition
    try:
        schema = json.loads(policy.output_schema)
    except json.JSONDecodeError as exc:
        return ViolationLog(
            rule_name="output.schema_invalid_definition",
            action=ViolationAction.WARN,
            severity=ViolationSeverity.MEDIUM,
            message=f"Policy output_schema is not valid JSON: {exc}",
            offending_content=policy.output_schema[:200],
            run_id=run_id,
            node_name=node_name,
        )

    # Step 2: parse LLM output
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return ViolationLog(
            rule_name="output.schema_not_json",
            action=policy.schema_action,
            severity=ViolationSeverity.MEDIUM,
            message=f"LLM output is not valid JSON: {exc}",
            offending_content=text[:200],
            run_id=run_id,
            node_name=node_name,
        )

    # Step 3: validate against schema
    if not _HAS_JSONSCHEMA:
        # Can't validate without the library — warn once
        return ViolationLog(
            rule_name="output.schema_validator_unavailable",
            action=ViolationAction.WARN,
            severity=ViolationSeverity.LOW,
            message="jsonschema package is not installed; schema validation skipped",
            offending_content="",
            run_id=run_id,
            node_name=node_name,
        )

    try:
        jsonschema.validate(instance=data, schema=schema)
    except jsonschema.ValidationError as exc:
        return ViolationLog(
            rule_name="output.schema_mismatch",
            action=policy.schema_action,
            severity=ViolationSeverity.MEDIUM,
            message=f"LLM output does not match required schema: {exc.message}",
            offending_content=text[:500],
            run_id=run_id,
            node_name=node_name,
        )
    except jsonschema.SchemaError as exc:
        return ViolationLog(
            rule_name="output.schema_invalid_definition",
            action=ViolationAction.WARN,
            severity=ViolationSeverity.MEDIUM,
            message=f"Invalid JSON Schema in policy: {exc.message}",
            offending_content=policy.output_schema[:200],
            run_id=run_id,
            node_name=node_name,
        )

    return None
