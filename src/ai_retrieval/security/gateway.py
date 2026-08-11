"""Fail-closed security boundary for outbound model disclosure."""

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Protocol

from ai_retrieval.domain.execution import ExecutionContext
from ai_retrieval.domain.failures import FailureCode, TypedFailure
from ai_retrieval.domain.immutable import FrozenMapping, freeze
from ai_retrieval.domain.model_routing import ModelCandidate, ModelOperation
from ai_retrieval.observability.telemetry import ObservabilityService, TelemetryEvent
from ai_retrieval.security.models import (
    AuthenticationResult,
    EncryptionMetadata,
    SecuredModelRequest,
    SecurityDecision,
    SecurityPolicy,
)


class ServiceAuthenticator(Protocol):
    async def authenticate(
        self, service_identity: str, provider_id: str, context: ExecutionContext
    ) -> AuthenticationResult: ...


class SecuredModelGateway(Protocol):
    async def invoke(self, request: SecuredModelRequest, context: ExecutionContext) -> object: ...


class SecurityPolicyRegistry:
    """Immutable-by-version policies resolved from the execution binding."""

    def __init__(self, policies: Sequence[SecurityPolicy] = ()) -> None:
        self._policies: dict[str, SecurityPolicy] = {}
        for policy in policies:
            self.register(policy)

    def register(self, policy: SecurityPolicy) -> None:
        existing = self._policies.get(policy.version)
        if existing is not None and existing != policy:
            raise ValueError(f"security policy version {policy.version!r} is immutable")
        self._policies[policy.version] = policy

    def resolve(self, version: str | None) -> SecurityPolicy | None:
        return self._policies.get(version) if version else None


class SecurityRejection(RuntimeError):
    def __init__(self, failure: TypedFailure) -> None:
        super().__init__(failure.message)
        self.failure = failure


class ProtectedModelInvoker:
    """Authenticates and applies every policy decision before invoking a provider."""

    def __init__(
        self,
        policies: SecurityPolicyRegistry,
        authenticator: ServiceAuthenticator,
        gateway: SecuredModelGateway,
        observability: ObservabilityService,
        clock=None,
    ) -> None:
        self._policies = policies
        self._authenticator = authenticator
        self._gateway = gateway
        self._observability = observability
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    async def invoke(
        self,
        operation: ModelOperation,
        candidate: ModelCandidate,
        payload: Mapping[str, object],
        service_identity: str,
        context: ExecutionContext,
    ) -> object:
        policy = self._policies.resolve(context.configuration.security_policy_version)
        if policy is None:
            await self._reject(context, None, service_identity, "policy_unavailable", FailureCode.SECURITY_POLICY_UNAVAILABLE)
        assert policy is not None
        if not policy.available:
            await self._reject(context, policy, service_identity, "policy_unavailable", FailureCode.SECURITY_POLICY_UNAVAILABLE)
        if not policy.valid:
            await self._reject(context, policy, service_identity, "policy_invalid", FailureCode.SECURITY_POLICY_INVALID)
        if policy.decision is not SecurityDecision.ALLOW:
            reason = "policy_indeterminate" if policy.decision is SecurityDecision.INDETERMINATE else "policy_denied"
            await self._reject(context, policy, service_identity, reason, FailureCode.SECURITY_POLICY_DENIED)
        if candidate.provider_id not in policy.allowed_provider_ids:
            await self._reject(context, policy, service_identity, "provider_denied", FailureCode.SECURITY_PROVIDER_DENIED)
        if not operation.data_classes.issubset(policy.allowed_data_classes):
            await self._reject(context, policy, service_identity, "data_class_denied", FailureCode.SECURITY_DATA_DENIED)
        if not policy.transport_encryption or not policy.persistence_encryption:
            await self._reject(context, policy, service_identity, "encryption_metadata_missing", FailureCode.SECURITY_PROTECTION_REQUIRED)

        authentication = await self._authenticator.authenticate(service_identity, candidate.provider_id, context)
        if not authentication.authenticated or authentication.service_identity != service_identity:
            await self._reject(
                context, policy, service_identity,
                authentication.reason_code or "authentication_failed",
                FailureCode.MODEL_AUTHENTICATION_FAILED,
            )

        masked = self._mask(payload, policy)
        frozen = freeze(masked)
        assert isinstance(frozen, FrozenMapping)
        request = SecuredModelRequest(
            operation, candidate, frozen, service_identity, policy.version,
            operation.data_classes,
            EncryptionMetadata(policy.transport_encryption, policy.persistence_encryption),
        )
        return await self._gateway.invoke(request, context)

    @staticmethod
    def _mask(value: object, policy: SecurityPolicy, path: tuple[str, ...] = ()) -> object:
        if isinstance(value, Mapping):
            masked: dict[str, object] = {}
            for key, item in value.items():
                current = path + (str(key),)
                dotted = ".".join(current)
                if str(key) in policy.sensitive_fields or dotted in policy.sensitive_fields:
                    masked[str(key)] = policy.masking_representation
                else:
                    masked[str(key)] = ProtectedModelInvoker._mask(item, policy, current)
            return masked
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return tuple(ProtectedModelInvoker._mask(item, policy, path) for item in value)
        return value

    async def _reject(
        self,
        context: ExecutionContext,
        policy: SecurityPolicy | None,
        service_identity: str,
        reason: str,
        code: FailureCode,
    ) -> None:
        redaction = policy.redaction_policy if policy else SecurityPolicy(
            "unavailable", False, False, SecurityDecision.DENY,
            frozenset(), frozenset(), frozenset(),
        ).redaction_policy
        details = freeze({
            "policy_version": policy.version if policy else "unavailable",
            "service_identity": service_identity or "unavailable",
            "decision": "rejected",
            "reason_code": reason,
        })
        assert isinstance(details, FrozenMapping)
        event = TelemetryEvent(
            str(context.correlation_id), self._clock(), "security_decision",
            "security", "rejected", context.configuration.reference.version, details,
        )
        try:
            await self._observability.emit(event, redaction)
        finally:
            raise SecurityRejection(TypedFailure(code, reason))
