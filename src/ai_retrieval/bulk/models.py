"""Durable bulk-work records and lightweight transport values."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from ai_retrieval.domain.configuration import ConfigurationReference


class ArtifactKind(StrEnum):
    PAYLOAD = "payload"
    RESULT = "result"


class WorkState(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    CHECKPOINTED = "checkpointed"
    COMPLETED = "completed"


class TerminalWorkItemState(StrEnum):
    SUCCEEDED = "succeeded"
    VALIDATION_FAILED = "validation-failed"
    POLICY_REJECTED = "policy-rejected"
    BUDGET_EXHAUSTED = "budget-exhausted"
    RETRY_EXHAUSTED = "retry-exhausted"
    WRITE_BACK_FAILED = "write-back-failed"
    CANCELLED = "cancelled"


class TerminalJobClassification(StrEnum):
    SUCCEEDED = "succeeded"
    PARTIALLY_SUCCEEDED = "partially-succeeded"
    FAILED = "failed"
    BUDGET_EXHAUSTED = "budget-exhausted"
    CANCELLED = "cancelled"


class JobTerminalCause(StrEnum):
    COMPLETED = "completed"
    BUDGET_EXHAUSTED = "budget-exhausted"
    CANCELLED = "cancelled"


class CheckpointStage(StrEnum):
    ACCEPTED = "accepted"
    MODEL_COMPLETED = "model_completed"
    RESULT_STORED = "result_stored"
    VALIDATED = "validated"
    COMPLETED = "completed"

    @property
    def order(self) -> int:
        return tuple(CheckpointStage).index(self)


class ClaimStatus(StrEnum):
    ACQUIRED = "acquired"
    ACTIVE_DUPLICATE = "active_duplicate"


class OutboxStatus(StrEnum):
    PENDING = "pending"
    PUBLISHED = "published"


def _require_text(name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must not be blank")


@dataclass(frozen=True)
class ObjectReference:
    reference: str
    content_hash: str
    kind: ArtifactKind
    size_bytes: int

    def __post_init__(self) -> None:
        _require_text("object reference", self.reference)
        _require_text("content hash", self.content_hash)
        if self.size_bytes < 0:
            raise ValueError("object size must be non-negative")


@dataclass(frozen=True)
class WorkSubmission:
    job_id: str
    item_id: str
    idempotency_key: str
    payload: bytes

    def __post_init__(self) -> None:
        _require_text("job_id", self.job_id)
        _require_text("item_id", self.item_id)
        _require_text("idempotency_key", self.idempotency_key)


@dataclass(frozen=True)
class WorkItemRecord:
    job_id: str
    item_id: str
    idempotency_key: str
    payload_reference: ObjectReference
    configuration: ConfigurationReference
    state: WorkState
    attempt_count: int = 0
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    result_reference: ObjectReference | None = None
    revision: int = 0

    def __post_init__(self) -> None:
        for name, value in (("job_id", self.job_id), ("item_id", self.item_id), ("idempotency_key", self.idempotency_key)):
            _require_text(name, value)
        if self.attempt_count < 0 or self.revision < 0:
            raise ValueError("attempt count and revision must be non-negative")
        active = self.lease_owner is not None or self.lease_expires_at is not None
        if active != (self.lease_owner is not None and self.lease_expires_at is not None):
            raise ValueError("lease owner and expiry must be present together")
        if self.state is WorkState.IN_PROGRESS and not active:
            raise ValueError("in-progress work requires a lease")
        if self.state is not WorkState.IN_PROGRESS and active:
            raise ValueError("only in-progress work may hold a lease")

    @property
    def configuration_version(self) -> str:
        return self.configuration.version


@dataclass(frozen=True)
class CheckpointRecord:
    job_id: str
    item_id: str
    idempotency_key: str
    completed_stage: CheckpointStage
    attempt_count: int
    outcome: str
    configuration: ConfigurationReference
    result_reference: ObjectReference | None
    recorded_at: datetime
    sequence: int

    def __post_init__(self) -> None:
        _require_text("checkpoint outcome", self.outcome)
        if self.attempt_count < 1 or self.sequence < 1:
            raise ValueError("checkpoint attempt and sequence must be positive")

    @property
    def configuration_version(self) -> str:
        return self.configuration.version


@dataclass(frozen=True)
class WorkNotification:
    event_id: str
    job_id: str
    item_id: str
    idempotency_key: str
    configuration_version: str
    payload_reference: str
    payload_hash: str
    aggregate_revision: int

    def __post_init__(self) -> None:
        for name, value in (
            ("event_id", self.event_id), ("job_id", self.job_id), ("item_id", self.item_id),
            ("idempotency_key", self.idempotency_key), ("configuration_version", self.configuration_version),
            ("payload_reference", self.payload_reference), ("payload_hash", self.payload_hash),
        ):
            _require_text(name, value)
        if self.aggregate_revision < 0:
            raise ValueError("aggregate revision must be non-negative")


@dataclass(frozen=True)
class OutboxRecord:
    event_id: str
    notification: WorkNotification
    status: OutboxStatus = OutboxStatus.PENDING
    published_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.status is OutboxStatus.PUBLISHED and self.published_at is None:
            raise ValueError("published outbox records require a timestamp")
        if self.status is OutboxStatus.PENDING and self.published_at is not None:
            raise ValueError("pending outbox records cannot have a publish timestamp")


@dataclass(frozen=True)
class WorkClaim:
    status: ClaimStatus
    item: WorkItemRecord
    latest_checkpoint: CheckpointRecord | None

    @property
    def acquired(self) -> bool:
        return self.status is ClaimStatus.ACQUIRED


@dataclass(frozen=True)
class ResumeState:
    item: WorkItemRecord
    latest_checkpoint: CheckpointRecord | None
    next_stage: CheckpointStage | None


@dataclass(frozen=True)
class BrokerDelivery:
    delivery_id: str
    notification: WorkNotification

    def __post_init__(self) -> None:
        _require_text("delivery_id", self.delivery_id)


@dataclass(frozen=True)
class TerminalWorkItemRecord:
    job_id: str
    item_id: str
    idempotency_key: str
    state: TerminalWorkItemState
    failure_code: str | None
    failure_details: str | None
    configuration: ConfigurationReference
    attempt_count: int
    result_reference: ObjectReference | None
    recorded_at: datetime

    def __post_init__(self) -> None:
        for name, value in (("job_id", self.job_id), ("item_id", self.item_id), ("idempotency_key", self.idempotency_key)):
            _require_text(name, value)
        if self.attempt_count < 1:
            raise ValueError("terminal work item attempt count must be positive")
        if self.recorded_at.tzinfo is None or self.recorded_at.utcoffset() is None:
            raise ValueError("terminal work item timestamp must be timezone-aware")
        if self.state is TerminalWorkItemState.SUCCEEDED and (self.failure_code is not None or self.failure_details is not None):
            raise ValueError("succeeded work cannot contain failure details")
        if self.state is not TerminalWorkItemState.SUCCEEDED and (not self.failure_code or not self.failure_code.strip()):
            raise ValueError("nonsuccess terminal work requires a failure code")

    @property
    def configuration_version(self) -> str:
        return self.configuration.version


@dataclass(frozen=True)
class DeadLetterEntry:
    job_id: str
    item_id: str
    idempotency_key: str
    failure_code: str
    failure_details: str | None
    attempt_count: int
    configuration: ConfigurationReference
    result_reference: ObjectReference | None
    persisted_at: datetime

    def __post_init__(self) -> None:
        for name, value in (
            ("job_id", self.job_id), ("item_id", self.item_id),
            ("idempotency_key", self.idempotency_key), ("failure_code", self.failure_code),
        ):
            _require_text(name, value)
        if self.attempt_count < 1:
            raise ValueError("dead-letter attempt count must be positive")
        if self.persisted_at.tzinfo is None or self.persisted_at.utcoffset() is None:
            raise ValueError("dead-letter timestamp must be timezone-aware")


@dataclass(frozen=True)
class PersistenceFailureOutcome:
    job_id: str
    item_id: str
    operation: str
    reason: str
    recorded_at: datetime

    def __post_init__(self) -> None:
        for name, value in (("job_id", self.job_id), ("item_id", self.item_id), ("operation", self.operation), ("reason", self.reason)):
            _require_text(name, value)


@dataclass(frozen=True)
class TerminalStateGroup:
    state: TerminalWorkItemState
    count: int
    item_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.count != len(self.item_ids) or self.count < 0:
            raise ValueError("terminal state count must equal identifier count")
        if any(not item_id.strip() for item_id in self.item_ids) or len(set(self.item_ids)) != len(self.item_ids):
            raise ValueError("terminal state identifiers must be unique and non-blank")


@dataclass(frozen=True)
class TerminalJobReport:
    job_id: str
    classification: TerminalJobClassification
    groups: tuple[TerminalStateGroup, ...]
    total_count: int
    terminal_cause: JobTerminalCause
    completed_at: datetime

    def __post_init__(self) -> None:
        _require_text("job_id", self.job_id)
        if self.total_count <= 0:
            raise ValueError("terminal job reports require at least one work item")
        if tuple(group.state for group in self.groups) != tuple(TerminalWorkItemState):
            raise ValueError("terminal job report must contain every state in deterministic order")
        identifiers = tuple(item_id for group in self.groups for item_id in group.item_ids)
        if self.total_count != len(identifiers) or len(set(identifiers)) != len(identifiers):
            raise ValueError("terminal report groups must form a complete disjoint partition")


@dataclass(frozen=True)
class BulkStateTelemetry:
    job_id: str
    counts_by_state: tuple[tuple[TerminalWorkItemState, int], ...]
    retry_count: int
    validation_failure_count: int
    dead_letter_count: int
    token_usage: int
    monetary_cost_minor_units: int
    elapsed_ms: int
    persistence_failure_count: int = 0

    def __post_init__(self) -> None:
        values = (
            self.retry_count, self.validation_failure_count, self.dead_letter_count,
            self.token_usage, self.monetary_cost_minor_units, self.elapsed_ms,
            self.persistence_failure_count,
        )
        if any(value < 0 for value in values):
            raise ValueError("bulk telemetry values must be non-negative")
        if tuple(state for state, _ in self.counts_by_state) != tuple(TerminalWorkItemState):
            raise ValueError("bulk telemetry must contain every terminal state in deterministic order")


@dataclass(frozen=True)
class WorkItemExecutionResult:
    state: TerminalWorkItemState
    failure_code: str | None = None
    failure_details: str | None = None
    result: bytes | None = None
    token_usage: int = 0
    monetary_cost_minor_units: int = 0

    def __post_init__(self) -> None:
        if self.token_usage < 0 or self.monetary_cost_minor_units < 0:
            raise ValueError("work item usage must be non-negative")
        if self.state is TerminalWorkItemState.SUCCEEDED and (self.failure_code is not None or self.failure_details is not None):
            raise ValueError("succeeded execution cannot contain failure details")
        if self.state is not TerminalWorkItemState.SUCCEEDED and (not self.failure_code or not self.failure_code.strip()):
            raise ValueError("nonsuccess execution requires a failure code")
