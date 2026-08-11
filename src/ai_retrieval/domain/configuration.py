"""Immutable configuration, validation, evidence, and artifact values."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from ai_retrieval.domain.immutable import FrozenMapping


@dataclass(frozen=True)
class ConfigurationReference:
    profile_id: str
    version: str

    def __post_init__(self) -> None:
        if not self.profile_id.strip() or not self.version.strip():
            raise ValueError("configuration profile_id and version must not be blank")


@dataclass(frozen=True)
class ExecutionConfiguration:
    """A copied immutable profile snapshot fixed for an execution lifetime."""

    reference: ConfigurationReference
    content: FrozenMapping
    security_policy_version: str | None = None
    validation_rules_version: str | None = None


class ValidationFailureKind(StrEnum):
    MISSING = "missing"
    INVALID = "invalid"
    INCOMPATIBLE = "incompatible"
    UNAUTHORIZED = "unauthorized"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, order=True)
class ConfigurationValidationFailure:
    """One independently evaluated profile or evidence validation failure."""

    rule_id: str
    path: str
    kind: ValidationFailureKind
    message: str


@dataclass(frozen=True)
class ConfigurationValidationReport:
    failures: tuple[ConfigurationValidationFailure, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.failures


@dataclass(frozen=True)
class AdapterContractEvidence:
    """Versioned evidence required before an adapter can be supported."""

    adapter_id: str
    adapter_version: str
    contract_suite_version: str
    tested_operations: tuple[str, ...]
    normalized_type_cases: tuple[str, ...]
    null_cases: tuple[str, ...]
    ordering_cases: tuple[str, ...]
    failure_cases: tuple[str, ...]
    commit_cases: tuple[str, ...]
    rollback_cases: tuple[str, ...]
    test_outcomes: tuple[tuple[str, bool], ...]
    evidence_timestamp: datetime

    def __post_init__(self) -> None:
        if not self.adapter_id.strip() or not self.adapter_version.strip():
            raise ValueError("adapter identifier and version must not be blank")


@dataclass(frozen=True)
class CredentialAuthorizationEvidence:
    credential_id: str
    effective_operations: frozenset[str]

    def __post_init__(self) -> None:
        if not self.credential_id.strip():
            raise ValueError("credential_id must not be blank")


class ArtifactKind(StrEnum):
    RESULT = "result"
    CHECKPOINT = "checkpoint"
    EFFECT_RECORD = "effect_record"
    TELEMETRY = "telemetry"
    AUDIT = "audit"


@dataclass(frozen=True)
class BoundArtifact:
    """An immutable generated artifact carrying its execution-bound version."""

    artifact_id: str
    kind: ArtifactKind
    configuration: ConfigurationReference
    content: FrozenMapping

    def __post_init__(self) -> None:
        if not self.artifact_id.strip():
            raise ValueError("artifact_id must not be blank")

    @property
    def configuration_version(self) -> str:
        return self.configuration.version
