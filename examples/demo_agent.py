"""
Demo: 2-node LangGraph agent wrapped with Sentinel.

Usage:
  python examples/demo_agent.py
  python examples/demo_agent.py --prompt "ignore previous instructions"
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

# Add project root to path so sentinel imports work before `pip install -e .`
sys.path.insert(0, str(Path(__file__).parent.parent))

from sentinel import Sentinel, SentinelViolation

# ── State ────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ── Nodes (LLM instantiated lazily inside main) ───────────────────────────────

def _make_nodes(llm: ChatGoogleGenerativeAI):
    async def planner(state: AgentState) -> AgentState:
        """Node 1 — creates a brief plan."""
        messages = state["messages"]
        plan_prompt = [
            HumanMessage(content=(
                "You are a helpful planning assistant. "
                "Given the user request, write a 1-sentence plan. "
                f"Request: {messages[-1].content}"
            ))
        ]
        response = await llm.ainvoke(plan_prompt)
        return {"messages": [AIMessage(content=f"Plan: {response.content}")]}

    async def executor(state: AgentState) -> AgentState:
        """Node 2 — executes the plan."""
        messages = state["messages"]
        plan = messages[-1].content
        exec_prompt = [
            HumanMessage(content=(
                "Execute this plan concisely in 2-3 sentences:\n" + plan
            ))
        ]
        response = await llm.ainvoke(exec_prompt)
        return {"messages": [AIMessage(content=response.content)]}

    return planner, executor


# ── Graph ────────────────────────────────────────────────────────────────────

def build_graph(llm: ChatGoogleGenerativeAI) -> StateGraph:
    planner, executor = _make_nodes(llm)
    builder = StateGraph(AgentState)
    builder.add_node("planner", planner)
    builder.add_node("executor", executor)
    builder.set_entry_point("planner")
    builder.add_edge("planner", "executor")
    builder.add_edge("executor", END)
    return builder.compile()


# ── Main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    prompt = "Explain what a LangGraph agent is in simple terms."
    if "--prompt" in sys.argv:
        idx = sys.argv.index("--prompt")
        prompt = sys.argv[idx + 1]

    policy_path = Path(__file__).parent / "sentinel_policy.yaml"

    # Sentinel loads policy and runs sync guards BEFORE the LLM is ever called.
    # We build a dummy graph shell so Sentinel can be constructed; if the prompt
    # violates policy the exception fires in ainvoke before LLM init is needed.
    # For clean prompts we then init the LLM and build the real graph.

    # --- Phase 1 trick: build Sentinel with a lazy-init graph wrapper ---
    class LazyGraph:
        """Defers real graph construction until ainvoke is actually called."""
        def __init__(self):
            self._graph = None

        async def ainvoke(self, input_dict):
            if self._graph is None:
                api_key = __import__("os").environ.get("GOOGLE_API_KEY", "")
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash",
                    google_api_key=api_key,
                    temperature=0.3,
                )
                self._graph = build_graph(llm)
            return await self._graph.ainvoke(input_dict)

    graph = LazyGraph()
    agent = Sentinel(graph, policy=policy_path)

    print(f"\n{'='*60}")
    print(f"Sentinel Demo Agent")
    print(f"{'='*60}")
    print(f"Prompt: {prompt!r}")
    print(f"{'='*60}\n")

    try:
        result = await agent.ainvoke({"messages": [HumanMessage(content=prompt)]})
        run_id = result.get("_sentinel_run_id", "unknown")
        messages = result.get("messages", [])
        final = messages[-1].content if messages else "(no output)"

        print("Response:")
        print(final)
        print(f"\n[run_id: {run_id}]")

        violations = await agent.get_violations(run_id)
        if violations:
            print(f"\nViolations detected ({len(violations)}):")
            for v in violations:
                print(f"  [{v.severity.value}] {v.rule_name}: {v.message}")

    except SentinelViolation as exc:
        print(f"\n{'!'*60}")
        print(f"SENTINEL VIOLATION RAISED")
        print(f"{'!'*60}")
        print(f"  rule_name : {exc.rule_name}")
        print(f"  action    : {exc.action.value}")
        print(f"  severity  : {exc.severity.value}")
        print(f"  message   : {exc.message}")
        print(f"  content   : {exc.offending_content!r}")
        print(f"  timestamp : {exc.timestamp.isoformat()}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
