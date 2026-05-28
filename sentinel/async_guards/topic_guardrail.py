from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from sentinel.policy import OutputPolicy
from sentinel.violation import ViolationLog, ViolationSeverity


class TopicResult(BaseModel):
    on_topic: bool
    detected_topic: str
    reason: str


async def check(
    output_text: str,
    policy: OutputPolicy,
    run_id: str = "",
    node_name: str = "output",
    _llm=None,  # injected in tests
) -> Optional[ViolationLog]:
    if not policy.topic_guardrail:
        return None

    allowed = ", ".join(policy.topic_guardrail)

    if _llm is None:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            _llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0)
        except Exception as exc:
            print(f"[sentinel] TopicGuardrail LLM init failed: {exc}", flush=True)
            return None

    try:
        judge = _llm.with_structured_output(TopicResult)
        prompt = (
            f"You are a topic classifier. Allowed topics: [{allowed}]. "
            f"Determine whether the following RESPONSE is on one of these allowed topics.\n\n"
            f"RESPONSE: {output_text}\n\n"
            f"Return on_topic=true if it relates to any allowed topic, "
            f"detected_topic (what topic you found or 'none'), and reason."
        )
        result: TopicResult = await judge.ainvoke(prompt)
    except Exception as exc:
        print(f"[sentinel] Topic guardrail check failed: {exc}", flush=True)
        return None

    if not result.on_topic:
        return ViolationLog(
            rule_name="output.topic_guardrail",
            action=policy.topic_action,
            severity=ViolationSeverity.MEDIUM,
            message=(
                f"Off-topic response: detected_topic={result.detected_topic!r}. "
                f"Allowed: [{allowed}]. Reason: {result.reason}"
            ),
            offending_content=output_text[:200],
            run_id=run_id,
            node_name=node_name,
        )

    return None
