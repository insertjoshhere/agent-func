"""Immutable records for idempotent write-back and effect recovery."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from ai_retrieval.bulk.models import CheckpointRecord, WorkClaim
from ai_retrieval.relational.models import ApprovedEffect


def _require_text(name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must not be blank")


class TransactionBoundary(StrEnum):
    SHARED = "shared"
    NON_SHARED = "non_shared"


class EffectEvidenceStatus(StrEnum):
    ABSENT = "absent"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    AMBIGUOUS = "ambiguous"


class EffectAttemptStatus(StrEnum):
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    AMBIGUOUS = "ambiguous"


class EffectRecoveryStatus(StrEnum):
    COMMITTED = "committed"
    RECOVERED = "recovered"
    RECONCILIATION_REQUIRED = "reconciliation_required"
    WRITE_BACK_FAILED = "write_back_failed"


@dataclass(frozen=True)
class EffectRecord:
    job_id: str
    item_id: str
    idempotency_key: str
    target_dataset: str
    row_scope_digest: str
    mutation_digest: str
    affected_rows: int
    transaction_outcome: str
    adapter_version: str
    committed_at: datetime


    def __post_init__(self) -> None:
        for name, value in (
            ("job_id", self.job_id), ("item_id", self.item_id),
            ("idempotency_key", self.idempotency_key), ("target_dataset", self.target_dataset),
            ("row_scope_digest", self.row_scope_digest), ("mutation_digest", self.mutation_digest),
            ("transaction_outcome", self.transaction_outcome), ("adapter_version", self.adapter_version),
        ):
            _require_text(name, value)
        if self.affected_rows < 0:
            raise ValueError("affected rows must be non-negative")
        if self.committed_at.tzinfo is None or self.committed_at.utcoffset() is None:
            raise ValueError("committed_at must be timezone-aware")


@dataclass(frozen=True)
class EffectRequest:
    claim: WorkClaim
    effect: ApprovedEffect
    target_dataset: str
    row_scope_digest: str
    mutation_digest: str
    boundary: TransactionBoundary

    def __post_init__(self) -> None:
        for name, value in (
            ("target_dataset", self.target_dataset),
            ("row_scope_digest", self.row_scope_digest),
            ("mutation_digest", self.mutation_digest),
        ):
            _require_text(name, value)
        if not self.claim.acquired:
            raise ValueError("an acquired work claim is required")
        if self.effect.effect_id != self.claim.item.idempotency_key:
            raise ValueError("effect identifier must equal the work idempotency key")


@dataclass(frozen=True)
class EffectAttempt:
    status: EffectAttemptStatus
    affected_rows: int = 0
    transaction_outcome: str = "unknown"
    adapter_version: str = "unknown"

    def __post_init__(self) -> None:
        if self.affected_rows < 0:
            raise ValueError("affected rows must be non-negative")
        _require_text("transaction outcome", self.transaction_outcome)
        _require_text("adapter version", self.adapter_version)


@dataclass(frozen=True)
class EffectEvidence:
    status: EffectEvidenceStatus
    record: EffectRecord | None = None

    def __post_init__(self) -> None:
        if (self.status is EffectEvidenceStatus.COMMITTED) != (self.record is not None):
            raise ValueError("committed evidence requires exactly one effect record")


@dataclass(frozen=True)
class ReconciliationRecord:
    job_id: str
    item_id: str
    idempotency_key: str
    target_dataset: str
    row_scope_digest: str
    mutation_digest: str
    reason: str
    recorded_at: datetime
    def __post_init__(self) -> None:
        for name, value in (
            ("job_id", self.job_id), ("item_id", self.item_id),
            ("idempotency_key", self.idempotency_key), ("target_dataset", self.target_dataset),
            ("row_scope_digest", self.row_scope_digest), ("mutation_digest", self.mutation_digest),
            ("reason", self.reason),
        ):
            _require_text(name, value)
        if self.recorded_at.tzinfo is None or self.recorded_at.utcoffset() is None:
            raise ValueError("recorded_at must be timezone-aware")


@dataclass(frozen=True)
class EffectRecoveryResult:
    status: EffectRecoveryStatus
    effect_record: EffectRecord | None = None
    checkpoint: CheckpointRecord | None = None
    reconciliation: ReconciliationRecord | None = None
    reason_code: str = "unspecified"

    def __post_init__(self) -> None:
        _require_text("effect recovery reason", self.reason_code)

    @property
    def committed(self) -> bool:
        return self.status in {EffectRecoveryStatus.COMMITTED, EffectRecoveryStatus.RECOVERED}
