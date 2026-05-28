"""Prometheus metrics for Sentinel — Tier 2d.

Exposes:
  sentinel_violations_total   Counter  (rule_name, action, severity, service)
  sentinel_budget_tokens       Gauge    (run_id, service)   — current run token usage
  sentinel_budget_cost_usd     Gauge    (run_id, service)   — current run cost
  sentinel_rate_limit_requests Gauge    (user_id, service)  — requests in last minute
  sentinel_rate_limit_tokens   Gauge    (user_id, service)  — tokens in last hour

When prometheus_client is not installed all public functions are no-ops so the
rest of the package remains importable without the optional dependency.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentinel.violation import ViolationLog

try:
    from prometheus_client import Counter, Gauge, REGISTRY

    _violations_total = Counter(
        "sentinel_violations_total",
        "Total policy violations detected by Sentinel",
        ["rule_name", "action", "severity", "service"],
        registry=REGISTRY,
    )

    _shadow_violations_total = Counter(
        "sentinel_shadow_violations_total",
        "Shadow policy violations observed but not enforced",
        ["rule_name", "action", "severity", "service"],
        registry=REGISTRY,
    )

    _budget_tokens = Gauge(
        "sentinel_budget_tokens",
        "Committed token count for a Sentinel run",
        ["run_id", "service"],
        registry=REGISTRY,
    )

    _budget_cost_usd = Gauge(
        "sentinel_budget_cost_usd",
        "Estimated cost (USD) committed for a Sentinel run",
        ["run_id", "service"],
        registry=REGISTRY,
    )

    _rate_requests = Gauge(
        "sentinel_rate_limit_requests",
        "Requests made in the current sliding window",
        ["user_id", "service"],
        registry=REGISTRY,
    )

    _rate_tokens = Gauge(
        "sentinel_rate_limit_tokens",
        "Tokens consumed in the current sliding window",
        ["user_id", "service"],
        registry=REGISTRY,
    )

    _HAS_PROMETHEUS = True

except ImportError:
    _HAS_PROMETHEUS = False


def record_violation(log: "ViolationLog", service: str = "sentinel") -> None:
    """Increment the appropriate violations counter with labels from `log`.

    Shadow violations go to sentinel_shadow_violations_total so they appear
    as a separate series in dashboards and don't pollute the enforced-violation
    count.
    """
    if not _HAS_PROMETHEUS:
        return
    counter = _shadow_violations_total if log.shadow else _violations_total
    counter.labels(
        rule_name=log.rule_name,
        action=log.action.value,
        severity=log.severity.value,
        service=service,
    ).inc()


def update_budget(run_id: str, tokens: int, cost_usd: float, service: str = "sentinel") -> None:
    """Set budget gauges for `run_id` to the latest committed values."""
    if not _HAS_PROMETHEUS:
        return
    _budget_tokens.labels(run_id=run_id, service=service).set(tokens)
    _budget_cost_usd.labels(run_id=run_id, service=service).set(cost_usd)


def update_rate_usage(
    user_id: str,
    requests_last_minute: int,
    tokens_last_hour: int,
    service: str = "sentinel",
) -> None:
    """Set rate-limit gauges for `user_id` to the current window values."""
    if not _HAS_PROMETHEUS:
        return
    _rate_requests.labels(user_id=user_id, service=service).set(requests_last_minute)
    _rate_tokens.labels(user_id=user_id, service=service).set(tokens_last_hour)


def start_metrics_server(port: int = 9100) -> None:
    """Start a standalone HTTP server exposing /metrics on `port`.

    Call once at application startup. Subsequent calls are silently ignored.
    """
    if not _HAS_PROMETHEUS:
        print(
            "[sentinel] prometheus_client not installed — metrics server not started. "
            "Install it with: pip install prometheus-client",
            flush=True,
        )
        return
    from prometheus_client import start_http_server

    try:
        start_http_server(port)
        print(f"[sentinel] Prometheus metrics server started on :{port}/metrics", flush=True)
    except OSError as exc:
        print(f"[sentinel] Could not start metrics server on :{port}: {exc}", flush=True)


def is_available() -> bool:
    """Return True when prometheus_client is installed."""
    return _HAS_PROMETHEUS
