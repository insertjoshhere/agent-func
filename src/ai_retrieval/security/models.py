"""Immutable security-policy and protected model-invocation values."""

from dataclasses import dataclass
from enum import StrEnum

from collections.abc import Mapping, Sequence

from ai_retrieval.domain.immutable import FrozenMapping
from ai_retrieval.domain.model_routing import ModelCandidate, ModelOperation


class SecurityDecision(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True)
class RedactionPolicy:
    sensitive_fields: frozenset[str] = frozenset()
    sensitive_values: frozenset[str] = frozenset()
    representation: str = "[REDACTED]"

    def __post_init__(self) -> None:
        if not self.representation:
            raise ValueError("redaction representation must not be empty")


@dataclass(frozen=True)
class SecurityPolicy:
    version: str
    available: bool
    valid: bool
    decision: SecurityDecision
    allowed_provider_ids: frozenset[str]
    allowed_data_classes: frozenset[str]
    sensitive_fields: frozenset[str]
    redaction_representation: str = "[REDACTED]"
    masking_representation: str = "[MASKED]"
    transport_encryption: str | None = None
    persistence_encryption: str | None = None

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("security policy version must not be blank")
        if not self.redaction_representation or not self.masking_representation:
            raise ValueError("security representations must not be empty")

    @property
    def permits_disclosure(self) -> bool:
        return self.available and self.valid and self.decision is SecurityDecision.ALLOW

    @property
    def redaction_policy(self) -> RedactionPolicy:
        return RedactionPolicy(self.sensitive_fields, frozenset(), self.redaction_representation)


@dataclass(frozen=True)
class AuthenticationResult:
    authenticated: bool
    service_identity: str
    reason_code: str


@dataclass(frozen=True)
class EncryptionMetadata:
    transport: str
    persistence: str

    def __post_init__(self) -> None:
        if not self.transport.strip() or not self.persistence.strip():
            raise ValueError("encryption metadata must not be blank")


@dataclass(frozen=True)
class SecuredModelRequest:
    operation: ModelOperation
    candidate: ModelCandidate
    payload: FrozenMapping
    service_identity: str
    policy_version: str
    data_classes: frozenset[str]
    encryption: EncryptionMetadata

    def __post_init__(self) -> None:
        if not self.service_identity.strip() or not self.policy_version.strip():
            raise ValueError("secured request identity and policy version must not be blank")
