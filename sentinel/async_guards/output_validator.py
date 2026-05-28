from __future__ import annotations

from typing import Optional

from better_profanity import profanity

from sentinel.policy import OutputPolicy
from sentinel.violation import ViolationLog, ViolationSeverity

# Load once at import time — avoids repeated global-state mutation on every call
profanity.load_censor_words()


async def validate(
    output_text: str,
    policy: OutputPolicy,
    run_id: str = "",
    node_name: str = "output",
) -> Optional[ViolationLog]:
    if policy.toxicity_check and profanity.contains_profanity(output_text):
        return ViolationLog(
            rule_name="output.toxicity",
            action=policy.toxicity_action,
            severity=ViolationSeverity.HIGH,
            message="Toxic content detected in output",
            offending_content=output_text[:200],
            run_id=run_id,
            node_name=node_name,
        )
    return None
