import asyncio
from datetime import datetime, timezone

import pytest

from ai_retrieval.domain.configuration import ConfigurationReference, ExecutionConfiguration
from ai_retrieval.domain.execution import CancellationContext, DeadlineContext, ExecutionContext, ExecutionPath
from ai_retrieval.domain.failures import FailureCode
from ai_retrieval.domain.identifiers import CorrelationId, ExecutionId
from ai_retrieval.domain.immutable import freeze
from ai_retrieval.domain.model_routing import ModelCandidate, ModelOperation
from ai_retrieval.domain.work import ModelWork
from ai_retrieval.observability import ObservabilityService, TelemetryEvent, serialize_redacted
from ai_retrieval.security import (
    AuthenticationResult, ProtectedModelInvoker, RedactionPolicy, SecurityDecision,
    SecurityPolicy, SecurityPolicyRegistry, SecurityRejection,
)


class Authenticator:
    def __init__(self, authenticated=True):
        self.authenticated = authenticated
        self.calls = []

    async def authenticate(self, identity, provider, context):
        self.calls.append((identity, provider))
        return AuthenticationResult(self.authenticated, identity, "allow" if self.authenticated else "bad_credentials")


class Gateway:
    def __init__(self):
        self.calls = []

    async def invoke(self, request, context):
        self.calls.append(request)
        return "ok"


class Sink:
    def __init__(self):
        self.events = []

    async def emit(self, event):
        self.events.append(event)


def context(policy_version="security-1"):
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    config = ExecutionConfiguration(
        ConfigurationReference("default", "config-1"), freeze({}), policy_version, "rules-1"
    )
    return ExecutionContext(
        ExecutionId("execution"), CorrelationId("correlation"), ExecutionPath.INTERACTIVE,
        config, DeadlineContext(now, now), CancellationContext("cancel", 0),
    )


def operation(data_classes=frozenset({"sensitive"})):
    return ModelOperation(
        "operation", "tenant", "request", ExecutionPath.INTERACTIVE,
        ModelWork("extract", "payload", ("1",), frozenset({"extract"})),
        10, data_classes=data_classes,
    )


def candidate(provider="provider-a"):
    return ModelCandidate(provider, provider, frozenset({"extract"}), frozenset({"sensitive"}), 1, 1, 10, 1.0)


def policy(**overrides):
    values = {
        "version": "security-1", "available": True, "valid": True,
        "decision": SecurityDecision.ALLOW,
        "allowed_provider_ids": frozenset({"provider-a"}),
        "allowed_data_classes": frozenset({"sensitive"}),
        "sensitive_fields": frozenset({"secret", "nested.token"}),
        "redaction_representation": "<redacted>", "masking_representation": "<masked>",
        "transport_encryption": "tls-1.3", "persistence_encryption": "kms-key-1",
    }
    values.update(overrides)
    return SecurityPolicy(**values)


def test_security_checks_authentication_masking_routing_and_encryption_before_provider_call():
    authentication, gateway, sink = Authenticator(), Gateway(), Sink()
    invoker = ProtectedModelInvoker(
        SecurityPolicyRegistry((policy(),)), authentication, gateway, ObservabilityService(sink)
    )

    result = asyncio.run(invoker.invoke(
        operation(), candidate(), {"secret": "raw", "nested": {"token": "raw", "safe": "yes"}},
        "model-service", context(),
    ))

    assert result == "ok"
    assert authentication.calls == [("model-service", "provider-a")]
    request = gateway.calls[0]
    assert request.payload["secret"] == "<masked>"
    assert request.payload["nested"]["token"] == "<masked>"
    assert request.payload["nested"]["safe"] == "yes"
    assert request.encryption.transport == "tls-1.3"
    assert request.policy_version == "security-1"


@pytest.mark.parametrize("configured_policy,expected", [
    (None, FailureCode.SECURITY_POLICY_UNAVAILABLE),
    (policy(available=False), FailureCode.SECURITY_POLICY_UNAVAILABLE),
    (policy(valid=False), FailureCode.SECURITY_POLICY_INVALID),
    (policy(decision=SecurityDecision.DENY), FailureCode.SECURITY_POLICY_DENIED),
    (policy(decision=SecurityDecision.INDETERMINATE), FailureCode.SECURITY_POLICY_DENIED),
    (policy(allowed_provider_ids=frozenset()), FailureCode.SECURITY_PROVIDER_DENIED),
    (policy(allowed_data_classes=frozenset()), FailureCode.SECURITY_DATA_DENIED),
    (policy(transport_encryption=None), FailureCode.SECURITY_PROTECTION_REQUIRED),
])
def test_non_allow_security_decisions_fail_closed_and_emit_correlated_redacted_audit(configured_policy, expected):
    authentication, gateway, sink = Authenticator(), Gateway(), Sink()
    registry = SecurityPolicyRegistry((configured_policy,)) if configured_policy else SecurityPolicyRegistry()
    invoker = ProtectedModelInvoker(registry, authentication, gateway, ObservabilityService(sink))

    with pytest.raises(SecurityRejection) as raised:
        asyncio.run(invoker.invoke(operation(), candidate(), {"secret": "never disclose"}, "model-service", context()))

    assert raised.value.failure.code is expected
    assert gateway.calls == []
    assert sink.events[0]["correlation_id"] == "correlation"
    assert sink.events[0]["configuration_version"] == "config-1"
    assert sink.events[0]["details"]["decision"] == "rejected"
    assert sink.events[0]["details"]["reason_code"]


def test_authentication_failure_prevents_external_submission():
    authentication, gateway, sink = Authenticator(False), Gateway(), Sink()
    invoker = ProtectedModelInvoker(
        SecurityPolicyRegistry((policy(),)), authentication, gateway, ObservabilityService(sink)
    )

    with pytest.raises(SecurityRejection) as raised:
        asyncio.run(invoker.invoke(operation(), candidate(), {}, "model-service", context()))

    assert raised.value.failure.code is FailureCode.MODEL_AUTHENTICATION_FAILED
    assert gateway.calls == []


def test_recursive_telemetry_redaction_retains_required_envelope_and_decision_fields():
    details = freeze({
        "principal": "service", "decision": "rejected", "reason_code": "policy_denied",
        "secret": "alpha", "nested": {"token": "beta", "values": ["alpha", "safe"]},
    })
    event = TelemetryEvent(
        "correlation", datetime(2025, 1, 1, tzinfo=timezone.utc), "security_decision",
        "security", "rejected", "config-1", details,
    )

    serialized = serialize_redacted(
        event, RedactionPolicy(frozenset({"secret", "nested.token"}), frozenset({"alpha"}), "<redacted>")
    )

    assert serialized["correlation_id"] == "correlation"
    assert serialized["details"]["decision"] == "rejected"
    assert serialized["details"]["secret"] == "<redacted>"
    assert serialized["details"]["nested"]["token"] == "<redacted>"
    assert serialized["details"]["nested"]["values"] == ("<redacted>", "safe")
