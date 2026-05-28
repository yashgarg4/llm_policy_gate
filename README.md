# Sentinel AI

**Policy-driven safety middleware for LangGraph agents — enforce compliance without touching agent logic.**

![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)
![Tests](https://img.shields.io/badge/Tests-279%20passing-brightgreen)

---

## What is this?

Sentinel is a declarative safety and compliance layer that wraps any LangGraph agent. You define a YAML policy — blocking patterns, PII rules, rate limits, budget caps, toxicity thresholds — and Sentinel enforces it at runtime before violations ever reach your users. It operates as a security sidecar: zero changes to your agent's internal logic, full coverage at the boundary.

Guards run in a two-speed pipeline. Sync guards execute _before_ the agent call and can hard-block execution. Async guards run _after_ the response as fire-and-forget tasks and never add latency to the hot path. Advanced features include multi-tenant policy routing, shadow mode for safe policy canary testing, and pluggable persistent violation stores.

---

## Features

### Input Guards
- **Block patterns** — regex-based blocklist applied to every user input before the graph runs. Case-insensitive, configurable list. Raises `SentinelViolation(BLOCK)` instantly.
- **PII detection & redaction** — scans input for email addresses, phone numbers, SSNs, and credit card numbers via [Presidio](https://github.com/microsoft/presidio). Configurable action: `REDACT` (replaces with `[REDACTED_TYPE]`) or `BLOCK`.
- **Semantic injection detection** — measures cosine similarity between the input and 36 canonical prompt-injection templates using TF-IDF char n-grams (sklearn). Falls back to Jaccard similarity when sklearn is not installed. Catches obfuscated attacks (extra spaces, character substitutions) that regex misses.
- **Token limit** — hard cap on input token count. Prevents prompt-stuffing attacks that aim to exhaust the context window.

### Runtime Safety
- **Budget gate** — enforces a per-run USD cost ceiling and token ceiling. Uses a peek/commit split: a pre-flight estimate checks the ceiling before the LLM call; actual token counts (from the LangChain callback) commit to the accumulator after. A run that exceeds the ceiling mid-stream is aborted immediately.
- **Rate limiter** — sliding-window rate limiter keyed by `user_id`. Two independent windows: requests per minute and tokens per hour. Thread-safe; uses a non-committing `peek()` variant for shadow mode so shadow observation never consumes real quota.
- **Circuit breaker** — detects two failure modes: a single graph node firing more than `max_node_repeats` times (infinite loop), and total LLM retries across the run exceeding `max_retries` (retry flood). Triggers an `ABORT` violation to stop the runaway agent.
- **Per-node LLM callback metering** — extends LangChain's `AsyncCallbackHandler` to intercept every LLM call inside the graph. Captures the actual token count from the LLM response (supports OpenAI, Anthropic, and Gemini token metadata formats) and commits it to the budget tracker. Raises `SentinelViolation` directly from inside the callback if a budget or circuit-breaker limit is hit mid-run.

### Output Guards *(async, never add latency)*
- **Toxicity check** — scans LLM output for profanity and toxic content using `better-profanity`. Runs as a fire-and-forget `asyncio.Task` after the response is returned, so it never holds up the caller.
- **Hallucination detection** — uses Gemini-as-judge to assess whether the response is grounded in the provided context. Configurable confidence threshold (0.0–1.0).
- **Topic guardrail** — Gemini classifies the response against an allowed-topics list. Flags responses that drift off-topic.
- **JSON schema validation** — validates structured LLM output against a JSON Schema defined in the policy. Three violation types: `schema_invalid_definition`, `schema_not_json`, `schema_mismatch`.

### Policy Management
- **Declarative YAML policy** — all guard thresholds, actions, and toggles live in a single YAML file validated against a Pydantic v2 schema. Rich validation errors on load.
- **Hot-reload** — the policy file is watched via `watchdog`. Edit and save while the process is running; the next call picks up the new policy with no restart.
- **Multi-policy routing** — supply a `policy_router(user_id, metadata) -> Optional[SentinelPolicy]` callable to route different users or tenants to different policies at runtime. Router exceptions fall back to the default policy safely.

### Advanced
- **Shadow mode** — supply a `shadow_router` to run a candidate policy in parallel with the live one. Shadow violations are logged with `shadow=True` but are never raised. Budget and rate-limit checks use non-committing `peek()` variants so shadow observation never consumes real quota. Use this to safely canary-test a stricter policy in production before enforcing it.
- **Persistent violation stores** — pluggable `ViolationStore` backends: in-memory (default), SQLite (`aiosqlite`, survives restarts), Redis (`redis.asyncio`, cross-process with configurable TTL). Write-through pattern keeps the sync code path unchanged while persisting violations in the background.

### Observability
- **OpenTelemetry tracing** — emits a `sentinel.violation` span per violation to any OTLP HTTP endpoint. Each span carries rule name, action, severity, run ID, node name, offending content, and a `sentinel.shadow` attribute.
- **Prometheus metrics** — six metrics covering violation counts, shadow violation counts, per-run budget usage, and per-user rate-limit window state. All metrics are no-ops when `prometheus_client` is not installed.
- **CLI** — `sentinel check --policy <file> --test-prompt <text>` validates a prompt offline. Exit code `0` = clean, `1` = blocking violation. Useful as a CI gate.

---

## Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │             Sentinel.ainvoke()               │
                    └─────────────────────┬────────────────────────┘
                                          │
                    ┌─────────────────────▼────────────────────────┐
                    │          SYNC GUARDS  (blocking)             │
                    │  ┌──────────────────────────────────────┐    │
                    │  │  Input Validator                     │    │
                    │  │  · Block patterns (regex)            │    │
                    │  │  · PII detection  (Presidio)         │    │
                    │  │  · Semantic injection (TF-IDF)       │    │
                    │  │  · Token limit check                 │    │
                    │  ├──────────────────────────────────────┤    │
                    │  │  Budget Gate  (peek → commit)        │    │
                    │  │  · Per-run cost ceiling (USD)        │    │
                    │  │  · Per-run token ceiling             │    │
                    │  ├──────────────────────────────────────┤    │
                    │  │  Rate Limiter  (sliding window)      │    │
                    │  │  · Requests / minute  per user_id   │    │
                    │  │  · Tokens / hour      per user_id   │    │
                    │  ├──────────────────────────────────────┤    │
                    │  │  Circuit Breaker                     │    │
                    │  │  · Node repeat detection             │    │
                    │  │  · Retry flood detection             │    │
                    │  └──────────────────────────────────────┘    │
                    └─────────────────────┬────────────────────────┘
                                          │  (unblocked)
                    ┌─────────────────────▼────────────────────────┐
                    │           LangGraph Agent Call               │
                    └─────────────────────┬────────────────────────┘
                                          │
                    ┌─────────────────────▼────────────────────────┐
                    │     ASYNC GUARDS  (fire-and-forget)          │
                    │  · Output Validator   (toxicity)             │
                    │  · Hallucination Detector  (Gemini judge)    │
                    │  · Topic Guardrail    (off-topic)            │
                    │  · JSON Schema Validator                     │
                    └──────────────────────────────────────────────┘
```

---

## Guards at a glance

| Guard | Phase | What it checks | Default action |
|---|---|---|---|
| Input Validator | Sync | Regex block patterns, PII entities, prompt injection | `BLOCK` / `REDACT` |
| Budget Gate | Sync | USD cost ceiling and token ceiling per run | `ABORT` |
| Rate Limiter | Sync | Requests/min and tokens/hour per `user_id` | `BLOCK` |
| Circuit Breaker | Sync | Repeated node calls and retry floods | `ABORT` |
| Output Validator | Async | Profanity / toxicity (`better-profanity`) | `FLAG` |
| Hallucination Detector | Async | Gemini-as-judge grounding check | `FLAG` |
| Topic Guardrail | Async | Off-topic response detection | `WARN` |
| Schema Validator | Async | JSON output matches declared schema | `FLAG` |

**ViolationAction** values: `BLOCK` · `ABORT` · `FLAG` · `REDACT` · `WARN`

**ViolationSeverity** values: `CRITICAL` · `HIGH` · `MEDIUM` · `LOW`

---

## Quick start

### Install

```bash
pip install -e .
```

### Wrap your agent (2 lines)

```python
from sentinel import Sentinel
from langchain_core.messages import HumanMessage

# graph is any compiled LangGraph StateGraph
agent = Sentinel(graph, policy="sentinel_policy.yaml")

result = await agent.ainvoke({"messages": [HumanMessage(content=user_input)]})
```

### Catch a violation

```python
from sentinel.violation import SentinelViolation

try:
    result = await agent.ainvoke({"messages": [HumanMessage(content=user_input)]})
except SentinelViolation as exc:
    print(exc.rule_name)   # e.g. "pii_detection"
    print(exc.action)      # ViolationAction.BLOCK
    print(exc.severity)    # ViolationSeverity.HIGH
    print(exc.run_id)      # UUID for this run
    print(exc.message)     # Human-readable description
```

### Retrieve violation logs

```python
result = await agent.ainvoke({"messages": [...]})
run_id = result["_sentinel_run_id"]

all_violations  = await agent.get_violations(run_id)
enforced_only   = await agent.get_violations(run_id, include_shadow=False)
```

### Streaming

```python
async for chunk in agent.astream({"messages": [...]}):
    process(chunk)
# All guards and shadow mode work identically in streaming
```

---

## Policy YAML reference

```yaml
service: my-agent                          # Arbitrary service identifier

input:
  max_tokens: 2048                         # Hard token ceiling on input
  block_patterns:                          # List of regex strings (case-insensitive)
    - "ignore (previous|all) instructions"
    - "system prompt"
  pii_detection: true                      # Requires: pip install -e ".[pii]"
  pii_action: REDACT                       # BLOCK | REDACT
  semantic_injection: true                 # Requires: pip install -e ".[semantic]"
  semantic_threshold: 0.85                 # Cosine similarity threshold (0.0–1.0)
  semantic_action: BLOCK

budget:
  max_cost_usd: 0.10                       # Max USD spend per run
  max_tokens_per_run: 4096                 # Max tokens consumed per run
  action: ABORT

circuit_breaker:
  max_node_repeats: 5                      # Abort if any node fires more than N times
  max_retries: 3                           # Abort if retry count exceeds N
  action: ABORT

rate_limit:
  enabled: true
  max_requests_per_minute: 20             # Per user_id
  max_tokens_per_hour: 100000             # Per user_id
  action: BLOCK

output:
  toxicity_check: true
  toxicity_action: FLAG
  topic_guardrail: true
  topic_action: WARN
  output_schema:                           # Optional JSON Schema for structured output
    type: object
    properties:
      answer: { type: string }
    required: [answer]
  schema_action: FLAG

hallucination:
  enabled: true
  threshold: 0.7                           # Gemini judge confidence threshold
  action: FLAG

audit:
  log_all: true                            # Log every run, not just violations
  tracely_endpoint: "https://tracely.example.com/otlp"
```

Policy files are **hot-reloaded** via `watchdog` — edit and save while the process is running; no restart required.

---

## Advanced features

### Multi-policy routing

Route different users or tenants to different policies at runtime.

```python
def my_router(user_id: str, metadata: dict):
    if metadata.get("tier") == "enterprise":
        return enterprise_policy   # SentinelPolicy instance
    return None  # Fall back to default policy

agent = Sentinel(graph, policy="base_policy.yaml", policy_router=my_router)
```

Exceptions from the router are caught and logged; execution falls back to the default policy safely.

---

### Shadow mode

Run a candidate policy in parallel, observe violations without ever blocking the user.

```python
agent = Sentinel(
    graph,
    policy="current_policy.yaml",
    shadow_router=lambda user_id, meta: candidate_policy,
)
```

- Shadow violations are logged with `shadow=True` and emitted to a dedicated Prometheus counter (`sentinel_shadow_violations_total`).
- Shadow mode uses `peek()` (non-committing) variants of the rate limiter and budget gate — shadow observation **never consumes real quota**.
- Nothing raised by the shadow policy is propagated to the caller.

---

### Persistent violation stores

```python
from sentinel.stores.memory import MemoryViolationStore      # default, no extras needed
from sentinel.stores.sqlite import SQLiteViolationStore      # pip install -e ".[sqlite]"
from sentinel.stores.redis_store import RedisViolationStore  # pip install -e ".[redis]"

agent = Sentinel(graph, policy="policy.yaml", store=SQLiteViolationStore("violations.db"))
```

| Store | Backend | Persistence |
|---|---|---|
| `MemoryViolationStore` | In-process dict | Current process lifetime |
| `SQLiteViolationStore` | `aiosqlite` | Survives restarts |
| `RedisViolationStore` | `redis.asyncio` (TTL-keyed) | Cross-process, distributed |

All stores follow a write-through pattern: synchronous dict write for immediate in-session reads, background `create_task` for persistence. `get_violations()` reads the in-memory dict for the current session and falls back to the store for historical runs.

---

## Observability

### OpenTelemetry

Sentinel emits one `sentinel.violation` span per violation to any OTLP HTTP endpoint (configured via `audit.tracely_endpoint` in the policy). Span attributes include rule name, action, severity, run ID, user ID, and the `sentinel.shadow` flag for shadow-mode violations.

### Prometheus

```python
# All metrics are no-ops when prometheus_client is not installed
sentinel_violations_total          # counter — labels: rule, action, severity, service
sentinel_shadow_violations_total   # counter — labels: rule, action, severity, service
sentinel_budget_cost_usd           # gauge   — current run cost
sentinel_budget_tokens             # gauge   — current run token count
sentinel_rate_limit_requests       # gauge   — sliding window request count
sentinel_rate_limit_tokens         # gauge   — sliding window token count
```

Install Prometheus support: `pip install -e ".[metrics]"`

---

## CLI

Test a prompt against a policy without running an agent — useful as a CI gate.

```bash
sentinel check --policy sentinel_policy.yaml --test-prompt "ignore previous instructions"
```

Exit codes: `0` = no blocking violation, `1` = blocking violation detected.

```yaml
# .github/workflows/ci.yml
- run: sentinel check --policy sentinel_policy.yaml --test-prompt "${{ env.TEST_PROMPT }}"
```

---

## Project structure

```
sentinel/
├── core.py                     # Sentinel class — main orchestrator
├── policy.py                   # Pydantic v2 schema + YAML loader
├── violation.py                # ViolationLog dataclass + SentinelViolation exception
├── callbacks.py                # AsyncCallbackHandler — per-node LLM token metering
├── telemetry.py                # OTel violation span emitter
├── metrics.py                  # Prometheus counters and gauges
├── watcher.py                  # Policy hot-reload via watchdog
├── sync_guards/
│   ├── input_validator.py      # Patterns, PII, semantic injection, token limit
│   ├── budget_gate.py          # Cost + token budget (peek / commit)
│   ├── rate_limiter.py         # Sliding window rate limiter (peek / commit)
│   ├── circuit_breaker.py      # Loop and retry detection
│   └── semantic_detector.py    # TF-IDF cosine + Jaccard fallback injection detector
├── async_guards/
│   ├── output_validator.py     # Toxicity (better-profanity)
│   ├── hallucination.py        # Gemini grounding judge
│   ├── topic_guardrail.py      # Off-topic detection
│   └── schema_validator.py     # JSON schema output validation
└── stores/
    ├── base.py                 # ViolationStore ABC
    ├── memory.py               # In-memory store
    ├── sqlite.py               # SQLite via aiosqlite
    └── redis_store.py          # Redis via redis.asyncio

cli/check.py                    # CLI entry point
tests/                          # 279 tests, 0 LLM calls (all mocked)
examples/
├── sentinel_policy.yaml        # Full example policy
└── demo_agent.py               # 2-node LangGraph demo
```

---

## Installation

```bash
# Base (LangGraph + core guards)
pip install -e .

# Optional extras — install only what you need
pip install -e ".[pii]"        # Presidio PII detection
pip install -e ".[semantic]"   # scikit-learn semantic injection detection
pip install -e ".[schema]"     # jsonschema structured output validation
pip install -e ".[metrics]"    # Prometheus client
pip install -e ".[sqlite]"     # SQLite persistent violation store
pip install -e ".[redis]"      # Redis persistent violation store

# Everything (development)
pip install -e ".[dev]"
```

**Requirements:** Python 3.11+, LangGraph, LangChain Core, Pydantic v2

---

## Running tests

```bash
pytest                          # run all 279 tests
pytest -x                       # stop on first failure
pytest tests/sync_guards/       # run a specific subsuite
pytest -v --tb=short            # verbose with short tracebacks
```

All tests are fully offline — every LLM call is mocked. No API keys required to run the test suite.
