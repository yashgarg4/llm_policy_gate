"""sentinel check — validate a prompt against a policy file without running an agent."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python cli/check.py` without an install
sys.path.insert(0, str(Path(__file__).parent.parent))

from sentinel.policy import load_policy
from sentinel.sync_guards.budget_gate import BudgetTracker
from sentinel.sync_guards.circuit_breaker import CircuitBreakerState
from sentinel.sync_guards.input_validator import validate as validate_input
from sentinel.violation import ViolationAction, ViolationLog


_BLOCKING = {ViolationAction.BLOCK, ViolationAction.ABORT}


def _run_checks(prompt: str, policy_path: Path) -> list[ViolationLog]:
    try:
        policy = load_policy(policy_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    violations: list[ViolationLog] = []
    run_id = "cli-check"

    v = validate_input(prompt, policy.input, run_id=run_id, node_name="cli")
    if v:
        violations.append(v)

    budget = BudgetTracker()
    bv = budget.check(
        run_id=run_id,
        estimated_tokens=max(len(prompt.split()), 1),
        model=policy.service,
        policy=policy.budget,
        node_name="cli",
    )
    if bv:
        violations.append(bv)

    cb = CircuitBreakerState()
    cv = cb.check(run_id=run_id, node_name="cli", policy=policy.circuit_breaker)
    if cv:
        violations.append(cv)

    return violations


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="sentinel check",
        description="Run Sentinel sync guards against a test prompt without invoking an agent.",
    )
    parser.add_argument("--policy", required=True, help="Path to sentinel_policy.yaml")
    parser.add_argument("--test-prompt", required=True, dest="prompt", help="Prompt text to evaluate")
    args = parser.parse_args(argv)

    policy_path = Path(args.policy)
    violations = _run_checks(args.prompt, policy_path)

    blocking = [v for v in violations if v.action in _BLOCKING]

    if not violations:
        print("PASS — no violations detected")
        sys.exit(0)

    print(f"FAIL — {len(violations)} violation(s) detected\n")
    for v in violations:
        status = "BLOCK" if v.action in _BLOCKING else "FLAG"
        print(f"  [{status}] [{v.severity.value}] {v.rule_name}")
        print(f"         message  : {v.message}")
        if v.offending_content:
            preview = v.offending_content[:80]
            print(f"         content  : {preview!r}")
        print()

    if blocking:
        sys.exit(1)
    else:
        # Warnings/flags only — exit 0 so CI can distinguish
        sys.exit(0)


if __name__ == "__main__":
    main()
