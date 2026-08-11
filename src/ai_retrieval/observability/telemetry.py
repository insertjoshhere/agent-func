"""Correlated telemetry envelopes with recursive policy-defined redaction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, TYPE_CHECKING

from ai_retrieval.domain.immutable import FrozenMapping, freeze

if TYPE_CHECKING:
    from ai_retrieval.security.models import RedactionPolicy


@dataclass(frozen=True)
class TelemetryEvent:
    correlation_id: str
    timestamp: datetime
    event_type: str
    component: str
    outcome: str
    configuration_version: str
    details: FrozenMapping = field(default_factory=lambda: FrozenMapping(()))

    def __post_init__(self) -> None:
        required = (
            self.correlation_id,
            self.event_type,
            self.component,
            self.outcome,
            self.configuration_version,
        )
        if any(not value.strip() for value in required):
            raise ValueError("telemetry envelope fields must not be blank")


class SerializedTelemetrySink(Protocol):
    async def emit(self, event: FrozenMapping) -> None: ...


def _redact(value: object, policy: RedactionPolicy, path: tuple[str, ...] = ()) -> object:
    if isinstance(value, (FrozenMapping, Mapping)):
        output: dict[str, object] = {}
        for key, item in value.items():
            current = () if not path and str(key) == "details" else path + (str(key),)
            dotted = ".".join(current)
            if str(key) in policy.sensitive_fields or dotted in policy.sensitive_fields:
                output[str(key)] = policy.representation
            else:
                output[str(key)] = _redact(item, policy, current)
        return output
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_redact(item, policy, path) for item in value)
    if isinstance(value, str) and value in policy.sensitive_values:
        return policy.representation
    return value


def serialize_redacted(event: TelemetryEvent, policy: RedactionPolicy) -> FrozenMapping:
    """Return an immutable envelope after redacting nested fields and known values."""

    raw = {
        "correlation_id": event.correlation_id,
        "timestamp": event.timestamp.isoformat(),
        "event_type": event.event_type,
        "component": event.component,
        "outcome": event.outcome,
        "configuration_version": event.configuration_version,
        "details": event.details,
    }
    result = freeze(_redact(raw, policy))
    assert isinstance(result, FrozenMapping)
    return result


class ObservabilityService:
    def __init__(self, sink: SerializedTelemetrySink) -> None:
        self._sink = sink

    async def emit(self, event: TelemetryEvent, policy: RedactionPolicy) -> None:
        await self._sink.emit(serialize_redacted(event, policy))
