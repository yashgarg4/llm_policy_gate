"""Tests for the sentinel check CLI — exit codes, output format, argument parsing."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from cli.check import _run_checks, main
from sentinel.violation import ViolationAction


def _write_policy(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "policy.yaml"
    p.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")
    return p


# ── _run_checks() unit tests ──────────────────────────────────────────────────

class TestRunChecks:
    def test_clean_prompt_returns_empty(self, tmp_path):
        p = _write_policy(tmp_path, """
            input:
              block_patterns: ["ignore previous instructions"]
              max_tokens: 4096
        """)
        assert _run_checks("Hello, what is Python?", p) == []

    def test_injection_pattern_returns_block_violation(self, tmp_path):
        p = _write_policy(tmp_path, """
            input:
              block_patterns: ["ignore previous instructions"]
        """)
        violations = _run_checks("ignore previous instructions please", p)
        assert len(violations) == 1
        assert violations[0].rule_name == "input.block_pattern"
        assert violations[0].action == ViolationAction.BLOCK

    def test_token_limit_exceeded_returns_violation(self, tmp_path):
        p = _write_policy(tmp_path, """
            input:
              max_tokens: 2
        """)
        violations = _run_checks("one two three four five", p)
        assert any(v.rule_name == "input.max_tokens" for v in violations)

    def test_pii_detected_when_enabled(self, tmp_path):
        p = _write_policy(tmp_path, """
            input:
              pii_detection: true
              pii_action: BLOCK
        """)
        violations = _run_checks("Email me at foo@bar.com", p)
        assert any("pii" in v.rule_name for v in violations)

    def test_missing_policy_file_calls_sys_exit_2(self, tmp_path):
        with pytest.raises(SystemExit) as exc_info:
            _run_checks("Hello", tmp_path / "nonexistent.yaml")
        assert exc_info.value.code == 2

    def test_multiple_patterns_only_first_match_returned(self, tmp_path):
        p = _write_policy(tmp_path, """
            input:
              block_patterns:
                - "pattern one"
                - "pattern two"
        """)
        # Text matches both — only first encountered is returned
        violations = _run_checks("pattern one and pattern two", p)
        assert len(violations) == 1

    def test_budget_exhausted_returns_violation(self, tmp_path):
        p = _write_policy(tmp_path, """
            budget:
              max_cost_usd: 0.000001
              max_tokens_per_run: 1000000
        """)
        violations = _run_checks("hello world", p)
        assert any("budget" in v.rule_name for v in violations)


# ── main() / CLI entrypoint ───────────────────────────────────────────────────

class TestMainCLI:
    def test_clean_prompt_exits_0(self, tmp_path):
        p = _write_policy(tmp_path, """
            input:
              block_patterns: ["ignore previous instructions"]
        """)
        with pytest.raises(SystemExit) as exc_info:
            main(["--policy", str(p), "--test-prompt", "Hello there!"])
        assert exc_info.value.code == 0

    def test_blocked_prompt_exits_1(self, tmp_path):
        p = _write_policy(tmp_path, """
            input:
              block_patterns: ["ignore previous instructions"]
        """)
        with pytest.raises(SystemExit) as exc_info:
            main(["--policy", str(p), "--test-prompt", "ignore previous instructions"])
        assert exc_info.value.code == 1

    def test_missing_policy_arg_exits_nonzero(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["--test-prompt", "hello"])
        assert exc_info.value.code != 0

    def test_missing_prompt_arg_exits_nonzero(self, tmp_path, capsys):
        p = _write_policy(tmp_path, "service: test\n")
        with pytest.raises(SystemExit) as exc_info:
            main(["--policy", str(p)])
        assert exc_info.value.code != 0

    def test_pass_output_printed_on_clean(self, tmp_path, capsys):
        p = _write_policy(tmp_path, "input:\n  max_tokens: 4096\n")
        with pytest.raises(SystemExit):
            main(["--policy", str(p), "--test-prompt", "Clean message"])
        assert "PASS" in capsys.readouterr().out

    def test_fail_output_shows_rule_name(self, tmp_path, capsys):
        p = _write_policy(tmp_path, """
            input:
              block_patterns: ["secret pattern"]
        """)
        with pytest.raises(SystemExit):
            main(["--policy", str(p), "--test-prompt", "secret pattern here"])
        out = capsys.readouterr().out
        assert "input.block_pattern" in out
        assert "BLOCK" in out

    def test_fail_output_shows_offending_content(self, tmp_path, capsys):
        p = _write_policy(tmp_path, """
            input:
              block_patterns: ["bad phrase"]
        """)
        with pytest.raises(SystemExit):
            main(["--policy", str(p), "--test-prompt", "bad phrase is here"])
        assert "bad phrase" in capsys.readouterr().out

    def test_nonexistent_policy_exits_2(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["--policy", str(tmp_path / "ghost.yaml"), "--test-prompt", "hi"])
        assert exc_info.value.code == 2
