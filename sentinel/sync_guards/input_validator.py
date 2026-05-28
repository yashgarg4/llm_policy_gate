from __future__ import annotations

import functools
import re
from typing import Optional

from sentinel.policy import InputPolicy, ViolationAction
from sentinel.sync_guards.semantic_detector import detect as detect_semantic
from sentinel.violation import ViolationLog, ViolationSeverity

# Regex patterns for common PII when presidio is unavailable
_FALLBACK_PII_PATTERNS: list[tuple[str, str]] = [
    (r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", "EMAIL_ADDRESS"),
    (r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "PHONE_NUMBER"),
    (r"\b\d{3}-\d{2}-\d{4}\b", "US_SSN"),
    (r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b", "CREDIT_CARD"),
]

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    _analyzer = AnalyzerEngine()
    _anonymizer = AnonymizerEngine()
    _HAS_PRESIDIO = True
except ImportError:
    _analyzer = None  # type: ignore[assignment]
    _anonymizer = None  # type: ignore[assignment]
    _HAS_PRESIDIO = False


@functools.lru_cache(maxsize=512)
def _compile_pattern(pattern: str) -> re.Pattern:
    """Compile and cache regex patterns so each unique string is only compiled once."""
    return re.compile(pattern, re.IGNORECASE)


def _count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text.split())


def _detect_pii_presidio(text: str) -> list[tuple[str, int, int, str]]:
    """Detect PII using Presidio AnalyzerEngine; returns (type, start, end, value) tuples."""
    results = _analyzer.analyze(text=text, language="en")
    return [(r.entity_type, r.start, r.end, text[r.start:r.end]) for r in results]


def _redact_presidio(text: str, entities: list[tuple[str, int, int, str]]) -> str:
    """Redact PII using Presidio AnonymizerEngine with per-type labels."""
    from presidio_anonymizer.entities import OperatorConfig, RecognizerResult

    recognizer_results = [
        RecognizerResult(entity_type=et, start=s, end=e, score=1.0)
        for et, s, e, _ in entities
    ]
    entity_types = {et for et, _, _, _ in entities}
    operators = {
        et: OperatorConfig("replace", {"new_value": f"[REDACTED_{et}]"})
        for et in entity_types
    }
    result = _anonymizer.anonymize(
        text=text,
        analyzer_results=recognizer_results,
        operators=operators,
    )
    return result.text


def _detect_pii_fallback(text: str) -> list[tuple[str, int, int, str]]:
    hits: list[tuple[str, int, int, str]] = []
    for pattern, label in _FALLBACK_PII_PATTERNS:
        for m in re.finditer(pattern, text):
            hits.append((label, m.start(), m.end(), m.group()))
    return hits


def _redact_fallback(text: str, entities: list[tuple[str, int, int, str]]) -> str:
    """Manual redaction — sorted in reverse order so offsets stay valid."""
    for entity_type, start, end, _ in sorted(entities, key=lambda x: -x[1]):
        text = text[:start] + f"[REDACTED_{entity_type}]" + text[end:]
    return text


def validate(
    text: str,
    policy: InputPolicy,
    run_id: str = "",
    node_name: str = "",
) -> Optional[ViolationLog]:
    if not text:
        return None

    # Check 1: token count
    token_count = _count_tokens(text)
    if token_count > policy.max_tokens:
        return ViolationLog(
            rule_name="input.max_tokens",
            action=ViolationAction.BLOCK,
            severity=ViolationSeverity.HIGH,
            message=f"Input exceeds max token limit: {token_count} > {policy.max_tokens}",
            offending_content=text[:200],
            run_id=run_id,
            node_name=node_name,
        )

    # Check 2: block patterns — compile once via lru_cache, save match object
    for pattern in policy.block_patterns:
        compiled = _compile_pattern(pattern)
        m = compiled.search(text)
        if m:
            return ViolationLog(
                rule_name="input.block_pattern",
                action=ViolationAction.BLOCK,
                severity=ViolationSeverity.CRITICAL,
                message=f"Input matched block pattern: {pattern!r}",
                offending_content=m.group(),
                run_id=run_id,
                node_name=node_name,
            )

    # Check 3: semantic injection detection (Tier 2b)
    semantic_violation = detect_semantic(text, policy, run_id=run_id, node_name=node_name)
    if semantic_violation:
        return semantic_violation

    # Check 4: PII detection
    if policy.pii_detection:
        entities = _detect_pii_presidio(text) if _HAS_PRESIDIO else _detect_pii_fallback(text)
        if entities:
            types = ", ".join(sorted(set(e[0] for e in entities)))
            if policy.pii_action == ViolationAction.BLOCK:
                return ViolationLog(
                    rule_name="input.pii_detected",
                    action=ViolationAction.BLOCK,
                    severity=ViolationSeverity.HIGH,
                    message=f"PII detected in input: {types}",
                    offending_content=entities[0][3],
                    run_id=run_id,
                    node_name=node_name,
                )
            elif policy.pii_action == ViolationAction.REDACT:
                redacted = (
                    _redact_presidio(text, entities)
                    if _HAS_PRESIDIO
                    else _redact_fallback(text, entities)
                )
                return ViolationLog(
                    rule_name="input.pii_redacted",
                    action=ViolationAction.REDACT,
                    severity=ViolationSeverity.MEDIUM,
                    message=f"PII redacted from input: {types}",
                    offending_content=redacted,
                    run_id=run_id,
                    node_name=node_name,
                )

    return None
