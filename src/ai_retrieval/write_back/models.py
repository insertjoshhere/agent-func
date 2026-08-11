"""Immutable values for exact batch write-back authorization."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from ai_retrieval.bulk.effect_models import EffectRecoveryResult, TransactionBoundary
from ai_retrieval.bulk.models import ObjectReference, WorkClaim
from ai_retrieval.relational.models import MUTATION_OPERATIONS, OperationKind
from ai_retrieval.validation.models import ValidationResult


def _text(name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must not be blank")


@dataclass(frozen=True)
class WriteBackScope:
    """Every value that must compare exactly at the mutation boundary."""

    job_id: str
    item_scope: frozenset[str]
    data_scope: str
    target_dataset: str
    row_scope: str
    columns: frozenset[str]
    mutation_type: OperationKind
    validation_rules_version: str
    configuration_version: str
    valid_from: datetime
    valid_until: datetime

    def __post_init__(self) -> None:
        for name, value in (
            ("job_id", self.job_id), ("data_scope", self.data_scope),
            ("target_dataset", self.target_dataset), ("row_scope", self.row_scope),
            ("validation_rules_version", self.validation_rules_version),
            ("configuration_version", self.configuration_version),
        ):
            _text(name, value)
        if not self.item_scope or any(not value.strip() for value in self.item_scope):
            raise ValueError("item scope must contain non-blank identifiers")
        if not self.columns or any(not value.strip() for value in self.columns):
            raise ValueError("columns must contain non-blank identifiers")
        if self.mutation_type not in MUTATION_OPERATIONS:
            raise ValueError("write-back mutation type must be a mutation operation")
        if any(value.tzinfo is None or value.utcoffset() is None for value in (self.valid_from, self.valid_until)):
            raise ValueError("approval validity timestamps must be timezone-aware")
        if self.valid_until <= self.valid_from:
            raise ValueError("approval validity interval must be positive")


@dataclass(frozen=True)
class WriteBackPolicy:
    version: str
    scope: WriteBackScope
    active: bool = True

    def __post_init__(self) -> None:
        _text("write-back policy version", self.version)


@dataclass(frozen=True)
class ApprovalRecord:
    reference: str
    authority: str
    policy_version: str
    scope: WriteBackScope
    revoked: bool = False

    def __post_init__(self) -> None:
        _text("approval reference", self.reference)
        _text("approval authority", self.authority)
        _text("approval policy version", self.policy_version)


@dataclass(frozen=True)
class AuthorizationSnapshot:
    policy: WriteBackPolicy | None
    approval: ApprovalRecord | None


@dataclass(frozen=True)
class WriteBackCommand:
    claim: WorkClaim
    validation: ValidationResult
    policy_version: str
    approval_reference: str
    data_scope: str
    target_dataset: str
    row_scope: str
    columns: frozenset[str]
    mutation_type: OperationKind
    parameters: tuple[tuple[str, object], ...]
    boundary: TransactionBoundary

    def __post_init__(self) -> None:
        if not self.claim.acquired:
            raise ValueError("an acquired work claim is required")
        for name, value in (
            ("policy_version", self.policy_version),
            ("approval_reference", self.approval_reference),
            ("data_scope", self.data_scope),
            ("target_dataset", self.target_dataset),
            ("row_scope", self.row_scope),
        ):
            _text(name, value)
        if not self.columns or any(not value.strip() for value in self.columns):
            raise ValueError("columns must contain non-blank identifiers")
        if self.mutation_type not in MUTATION_OPERATIONS:
            raise ValueError("write-back mutation type must be a mutation operation")
        names = tuple(name for name, _ in self.parameters)
        if any(not name.strip() for name in names) or len(set(names)) != len(names):
            raise ValueError("mutation parameter names must be unique and non-blank")


class WriteBackExecutionStatus(StrEnum):
    VALIDATION_REJECTED = "validation_rejected"
    PERSISTED = "persisted"
    APPROVAL_REJECTED = "approval_rejected"
    COMMITTED = "committed"
    RECOVERED = "recovered"
    RECONCILIATION_REQUIRED = "reconciliation_required"
    WRITE_BACK_FAILED = "write_back_failed"


class WriteBackAuditOutcome(StrEnum):
    COMMIT = "commit"
    ROLLBACK = "rollback"
    RECONCILIATION = "reconciliation"
    RECOVERY_DUPLICATE = "recovery_duplicate"
    WITHHELD = "withheld"
    FAILURE = "failure"


@dataclass(frozen=True)
class WriteBackAuditEvent:
    correlation_id: str
    job_id: str
    item_id: str
    idempotency_key: str
    approval_reference: str
    approval_authority: str
    policy_version: str
    target_dataset: str
    row_scope_digest: str
    effect_digest: str
    columns: tuple[str, ...]
    mutation_type: str
    validation_rules_version: str
    configuration_version: str
    outcome: WriteBackAuditOutcome
    timestamp: datetime
    reason_codes: tuple[str, ...]
    redacted_details: str = "[REDACTED]"

    def __post_init__(self) -> None:
        required = (
            self.correlation_id, self.job_id, self.item_id, self.idempotency_key,
            self.approval_reference, self.approval_authority, self.policy_version,
            self.target_dataset, self.row_scope_digest, self.effect_digest,
            self.mutation_type, self.validation_rules_version,
            self.configuration_version, self.redacted_details,
        )
        if any(not value.strip() for value in required):
            raise ValueError("write-back audit fields must not be blank")
        if not self.columns or any(not value.strip() for value in self.columns):
            raise ValueError("write-back audit columns must not be blank")
        if not self.reason_codes or any(not value.strip() for value in self.reason_codes):
            raise ValueError("write-back audit reason codes must not be blank")
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("write-back audit timestamp must be timezone-aware")


@dataclass(frozen=True)
class WriteBackExecutionResult:
    status: WriteBackExecutionStatus
    reason_codes: tuple[str, ...] = ()
    output_reference: ObjectReference | None = None
    effect: EffectRecoveryResult | None = None

    @property
    def mutated(self) -> bool:
        return self.status in {WriteBackExecutionStatus.COMMITTED, WriteBackExecutionStatus.RECOVERED}
