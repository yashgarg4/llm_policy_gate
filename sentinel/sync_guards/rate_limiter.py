"""Cross-run sliding-window rate limiter — per user_id, across multiple ainvoke() calls.

Two independent windows:
  - requests per minute  (short, prevents burst abuse)
  - tokens per hour      (long, enforces cost envelope)
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Optional

from sentinel.policy import RateLimitPolicy
from sentinel.violation import ViolationLog, ViolationSeverity


class RateLimiter:
    """Thread-safe sliding-window rate limiter keyed by user_id."""

    _REQ_WINDOW_S = 60.0      # 1 minute
    _TOKEN_WINDOW_S = 3600.0  # 1 hour

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # user_id → deque of arrival timestamps (for request counting)
        self._req_ts: dict[str, deque[float]] = defaultdict(deque)
        # user_id → deque of (timestamp, token_count) pairs
        self._tok_ts: dict[str, deque[tuple[float, int]]] = defaultdict(deque)

    def check(
        self,
        user_id: str,
        tokens: int,
        policy: RateLimitPolicy,
        run_id: str = "",
        node_name: str = "",
    ) -> Optional[ViolationLog]:
        """Check and, if within limits, commit this request.

        Returns a ViolationLog if a limit is exceeded (nothing is committed
        when a violation is returned).
        """
        if not policy.enabled:
            return None

        now = time.monotonic()

        with self._lock:
            # Evict expired entries first so counts reflect the live window only
            req_q = self._req_ts[user_id]
            while req_q and now - req_q[0] > self._REQ_WINDOW_S:
                req_q.popleft()

            tok_q = self._tok_ts[user_id]
            while tok_q and now - tok_q[0][0] > self._TOKEN_WINDOW_S:
                tok_q.popleft()

            # Check request-rate limit
            if len(req_q) >= policy.max_requests_per_minute:
                return ViolationLog(
                    rule_name="rate_limit.requests_per_minute",
                    action=policy.action,
                    severity=ViolationSeverity.HIGH,
                    message=(
                        f"Request rate limit exceeded for user '{user_id}': "
                        f"{len(req_q)} requests in the last minute "
                        f"(limit: {policy.max_requests_per_minute})"
                    ),
                    offending_content=f"user_id={user_id}",
                    run_id=run_id,
                    node_name=node_name,
                )

            # Check token-rate limit
            tokens_in_window = sum(t for _, t in tok_q)
            if tokens_in_window + tokens > policy.max_tokens_per_hour:
                return ViolationLog(
                    rule_name="rate_limit.tokens_per_hour",
                    action=policy.action,
                    severity=ViolationSeverity.HIGH,
                    message=(
                        f"Token rate limit exceeded for user '{user_id}': "
                        f"{tokens_in_window + tokens:,} tokens in the last hour "
                        f"(limit: {policy.max_tokens_per_hour:,})"
                    ),
                    offending_content=f"user_id={user_id}",
                    run_id=run_id,
                    node_name=node_name,
                )

            # Commit — only reached when both checks pass
            req_q.append(now)
            tok_q.append((now, tokens))

        return None

    def peek(
        self,
        user_id: str,
        tokens: int,
        policy: RateLimitPolicy,
        run_id: str = "",
        node_name: str = "",
    ) -> Optional[ViolationLog]:
        """Read-only rate-limit check — does not commit the request to the window.

        Used by shadow guards so that observing a request never consumes quota.
        """
        if not policy.enabled:
            return None

        now = time.monotonic()

        with self._lock:
            req_q = self._req_ts[user_id]
            while req_q and now - req_q[0] > self._REQ_WINDOW_S:
                req_q.popleft()

            tok_q = self._tok_ts[user_id]
            while tok_q and now - tok_q[0][0] > self._TOKEN_WINDOW_S:
                tok_q.popleft()

            if len(req_q) >= policy.max_requests_per_minute:
                return ViolationLog(
                    rule_name="rate_limit.requests_per_minute",
                    action=policy.action,
                    severity=ViolationSeverity.HIGH,
                    message=(
                        f"Request rate limit exceeded for user '{user_id}': "
                        f"{len(req_q)} requests in the last minute "
                        f"(limit: {policy.max_requests_per_minute})"
                    ),
                    offending_content=f"user_id={user_id}",
                    run_id=run_id,
                    node_name=node_name,
                )

            tokens_in_window = sum(t for _, t in tok_q)
            if tokens_in_window + tokens > policy.max_tokens_per_hour:
                return ViolationLog(
                    rule_name="rate_limit.tokens_per_hour",
                    action=policy.action,
                    severity=ViolationSeverity.HIGH,
                    message=(
                        f"Token rate limit exceeded for user '{user_id}': "
                        f"{tokens_in_window + tokens:,} tokens in the last hour "
                        f"(limit: {policy.max_tokens_per_hour:,})"
                    ),
                    offending_content=f"user_id={user_id}",
                    run_id=run_id,
                    node_name=node_name,
                )

        return None

    def get_usage(self, user_id: str) -> dict[str, int]:
        """Return current window usage for the given user_id (for tests / dashboards)."""
        now = time.monotonic()
        with self._lock:
            req_q = self._req_ts[user_id]
            tok_q = self._tok_ts[user_id]
            reqs = sum(1 for t in req_q if now - t <= self._REQ_WINDOW_S)
            toks = sum(c for ts, c in tok_q if now - ts <= self._TOKEN_WINDOW_S)
        return {"requests_last_minute": reqs, "tokens_last_hour": toks}

    def reset(self, user_id: str) -> None:
        with self._lock:
            self._req_ts.pop(user_id, None)
            self._tok_ts.pop(user_id, None)
