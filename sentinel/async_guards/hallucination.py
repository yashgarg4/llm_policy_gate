from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from sentinel.policy import HallucinationPolicy
from sentinel.violation import ViolationLog, ViolationSeverity


class GroundingResult(BaseModel):
    grounded: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class HallucinationJudge:
    def __init__(self, model: str = "gemini-2.0-flash") -> None:
        from langchain_google_genai import ChatGoogleGenerativeAI
        self._llm = ChatGoogleGenerativeAI(model=model, temperature=0.0)
        self._judge = self._llm.with_structured_output(GroundingResult)

    async def judge(self, query: str, response: str, context: str) -> GroundingResult:
        prompt = (
            "You are a grounding evaluator. "
            "Determine whether the RESPONSE is factually grounded in the CONTEXT "
            "and accurately answers the QUERY. "
            "Return grounded=true only if all key claims are supported by the context.\n\n"
            f"QUERY: {query}\n\nCONTEXT: {context}\n\nRESPONSE: {response}\n\n"
            "Evaluate and respond with grounded, confidence (0.0–1.0), and reason."
        )
        return await self._judge.ainvoke(prompt)


async def check(
    query: str,
    response: str,
    context: str,
    policy: HallucinationPolicy,
    run_id: str = "",
    node_name: str = "output",
    judge: Optional[HallucinationJudge] = None,
) -> Optional[ViolationLog]:
    if not policy.enabled:
        return None

    if judge is None:
        try:
            judge = HallucinationJudge()
        except Exception as exc:
            print(f"[sentinel] HallucinationJudge init failed: {exc}", flush=True)
            return None

    try:
        result = await judge.judge(query=query, response=response, context=context)
    except Exception as exc:
        print(f"[sentinel] Hallucination check failed: {exc}", flush=True)
        return None

    if not result.grounded or result.confidence < policy.threshold:
        return ViolationLog(
            rule_name="hallucination.low_grounding",
            action=policy.action,
            severity=ViolationSeverity.MEDIUM,
            message=(
                f"Hallucination risk: grounded={result.grounded}, "
                f"confidence={result.confidence:.2f} < threshold={policy.threshold:.2f}. "
                f"Reason: {result.reason}"
            ),
            offending_content=response[:200],
            run_id=run_id,
            node_name=node_name,
        )

    return None
