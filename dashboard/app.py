"""Sentinel violation dashboard — reads from Tracely's SQLite DB (sync)."""
from __future__ import annotations

import json
import os
import sqlite3
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import streamlit as st

# Config 

_DEFAULT_DB = Path(
    os.getenv(
        "TRACELY_DB_PATH",
        str(Path(__file__).parent.parent.parent / "meridian" / "meridian-server" / "meridian.db"),
    )
)

st.set_page_config(
    page_title="Sentinel — Violation Dashboard",
    page_icon="🛡️",
    layout="wide",
)


# DB helpers (sync sqlite3 — no event-loop overhead in Streamlit) 

def _fetch_violations(db_path: Path, since_ts: float) -> list[dict]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT s.run_id, s.name, s.start_time, s.attributes, r.service_name
            FROM spans s
            JOIN runs r ON r.id = s.run_id
            WHERE s.name = 'sentinel.violation'
              AND s.start_time >= ?
            ORDER BY s.start_time DESC
            """,
            (since_ts,),
        ).fetchall()

    result = []
    for row in rows:
        attrs = json.loads(row["attributes"] or "{}")
        result.append({
            "run_id":           row["run_id"],
            "service":          attrs.get("sentinel.service", row["service_name"] or "—"),
            "rule_name":        attrs.get("sentinel.rule_name", "—"),
            "action":           attrs.get("sentinel.action", "—"),
            "severity":         attrs.get("sentinel.severity", "—"),
            "message":          attrs.get("sentinel.message", "—"),
            "offending_content":attrs.get("sentinel.offending_content", ""),
            "node_name":        attrs.get("sentinel.node_name", "—"),
            "timestamp":        datetime.fromtimestamp(row["start_time"], tz=timezone.utc),
        })
    return result


def _fetch_hourly(db_path: Path, since_ts: float) -> list[dict]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT
                strftime('%Y-%m-%dT%H:00', datetime(start_time, 'unixepoch')) AS hour,
                COUNT(*) AS count
            FROM spans
            WHERE name = 'sentinel.violation'
              AND start_time >= ?
            GROUP BY hour
            ORDER BY hour
            """,
            (since_ts,),
        ).fetchall()
    return [dict(row) for row in rows]


# Sidebar 

with st.sidebar:
    st.title("🛡️ Sentinel")
    st.caption("Policy Violation Dashboard")

    db_path_input = st.text_input(
        "Tracely DB path",
        value=str(_DEFAULT_DB),
        help="Absolute path to meridian.db",
    )
    db_path = Path(db_path_input)

    window_h = st.slider("Time window (hours)", min_value=1, max_value=168, value=24, step=1)
    auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)

    if auto_refresh:
        import time
        time.sleep(30)
        st.rerun()

    if st.button("🔄 Refresh now"):
        st.rerun()

# ── Load data ─────────────────────────────────────────────────────────────────

since_ts = (datetime.now(timezone.utc) - timedelta(hours=window_h)).timestamp()

if not db_path.exists():
    st.error(f"Database not found at `{db_path}`. Update the path in the sidebar.")
    st.stop()

try:
    violations = _fetch_violations(db_path, since_ts)
    hourly = _fetch_hourly(db_path, since_ts)
except sqlite3.OperationalError as exc:
    st.error(f"Database error: {exc}")
    st.stop()

# ── KPI row ───────────────────────────────────────────────────────────────────

total = len(violations)
blocked = sum(1 for v in violations if v["action"] in ("BLOCK", "ABORT"))
flagged = sum(1 for v in violations if v["action"] == "FLAG")
block_rate = f"{blocked / total * 100:.1f}%" if total else "—"
flag_rate  = f"{flagged / total * 100:.1f}%" if total else "—"
rule_counts = Counter(v["rule_name"] for v in violations)
top_rule = rule_counts.most_common(1)[0][0] if rule_counts else "—"

st.title("Sentinel Violation Dashboard")
st.caption(f"Last {window_h}h  ·  {total} violation{'s' if total != 1 else ''} total")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total violations", total)
c2.metric("Block rate", block_rate, help="BLOCK or ABORT as % of total")
c3.metric("Flag rate", flag_rate,  help="FLAG actions as % of total")
c4.metric("Top rule", top_rule)

st.divider()

# ── Time series ───────────────────────────────────────────────────────────────

st.subheader("Violations per hour")
if hourly:
    import pandas as pd
    df_h = pd.DataFrame(hourly)
    df_h["hour"] = pd.to_datetime(df_h["hour"])
    st.bar_chart(df_h.set_index("hour").rename(columns={"count": "violations"})["violations"])
else:
    st.info("No violation spans in the selected window.")

st.divider()

# ── Rule breakdown ────────────────────────────────────────────────────────────

if rule_counts:
    import pandas as pd
    st.subheader("Rule breakdown")
    df_r = pd.DataFrame(rule_counts.most_common(), columns=["rule", "count"]).set_index("rule")
    st.bar_chart(df_r["count"], horizontal=True)

st.divider()

# ── Recent violations table ───────────────────────────────────────────────────

st.subheader("Recent violations")

_SEV_ICON = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}
_ACT_ICON = {"BLOCK": "🚫", "ABORT": "⛔", "FLAG": "🚩", "REDACT": "✂️", "WARN": "⚠️"}

if not violations:
    st.info("No violations in this window.")
else:
    for i, v in enumerate(violations):
        sev_icon = _SEV_ICON.get(v["severity"], "⚪")
        act_icon = _ACT_ICON.get(v["action"], "")
        ts = v["timestamp"].strftime("%Y-%m-%d %H:%M:%S UTC")
        preview = (v["offending_content"] or v["message"])[:80]
        label = f"{sev_icon} {ts}  ·  {act_icon} {v['action']}  ·  `{v['rule_name']}`  ·  {preview!r}"

        with st.expander(label, expanded=(i == 0)):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Rule:** `{v['rule_name']}`")
                st.markdown(f"**Action:** {act_icon} `{v['action']}`")
                st.markdown(f"**Severity:** {sev_icon} `{v['severity']}`")
                st.markdown(f"**Node:** `{v['node_name']}`")
            with col2:
                st.markdown(f"**Service:** `{v['service']}`")
                st.markdown(f"**Run ID:** `{v['run_id'][:16]}…`")
                st.markdown(f"**Timestamp:** `{ts}`")

            st.markdown("**Message:**")
            st.info(v["message"])

            if v["offending_content"]:
                st.markdown("**Offending content:**")
                st.code(v["offending_content"], language=None)
