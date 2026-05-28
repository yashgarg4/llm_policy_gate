from __future__ import annotations

import threading

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from sentinel.violation import ViolationLog

_lock = threading.Lock()
_providers: dict[str, TracerProvider] = {}


def _get_tracer(endpoint: str, service_name: str) -> trace.Tracer:
    key = f"{endpoint}::{service_name}"
    with _lock:
        if key not in _providers:
            resource = Resource.create({"service.name": service_name})
            # SimpleSpanProcessor exports each span inline when it ends —
            # safe even when the process is about to raise and exit.
            exporter = OTLPSpanExporter(endpoint=endpoint)
            provider = TracerProvider(resource=resource)
            provider.add_span_processor(SimpleSpanProcessor(exporter))
            _providers[key] = provider
        return _providers[key].get_tracer("sentinel")


def _emit(violation: ViolationLog, endpoint: str, service_name: str) -> None:
    try:
        tracer = _get_tracer(endpoint, service_name)
        current_span = trace.get_current_span()
        ctx = trace.set_span_in_context(current_span) if current_span.is_recording() else None

        with tracer.start_as_current_span(
            "sentinel.violation",
            context=ctx,
            kind=trace.SpanKind.INTERNAL,
        ) as span:
            span.set_attribute("sentinel.rule_name", violation.rule_name)
            span.set_attribute("sentinel.action", violation.action.value)
            span.set_attribute("sentinel.severity", violation.severity.value)
            span.set_attribute("sentinel.service", service_name)
            span.set_attribute("sentinel.run_id", violation.run_id)
            span.set_attribute("sentinel.node_name", violation.node_name)
            span.set_attribute("sentinel.message", violation.message)
            if violation.offending_content:
                span.set_attribute(
                    "sentinel.offending_content",
                    violation.offending_content[:500],
                )
            span.set_attribute("sentinel.timestamp", violation.timestamp.isoformat())
            span.set_attribute("sentinel.shadow", violation.shadow)
    except Exception as exc:
        print(f"[sentinel] telemetry emit failed: {exc}", flush=True)


def emit_violation(
    violation: ViolationLog,
    tracely_endpoint: str,
    service_name: str = "sentinel",
    blocking: bool = False,
) -> None:
    """Emit a sentinel.violation span to Tracely.

    blocking=True: runs synchronously in the caller's thread (use before raising).
    blocking=False: runs in a daemon thread (use for fire-and-forget async violations).
    """
    if blocking:
        _emit(violation, tracely_endpoint, service_name)
    else:
        threading.Thread(
            target=_emit,
            args=(violation, tracely_endpoint, service_name),
            daemon=True,
        ).start()


def shutdown_all() -> None:
    with _lock:
        for provider in _providers.values():
            try:
                provider.shutdown()
            except Exception:
                pass
        _providers.clear()
