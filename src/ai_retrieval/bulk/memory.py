"""Thread-safe transactional in-memory adapters for durable bulk contracts."""

from dataclasses import replace
from datetime import datetime, timedelta
from hashlib import sha256
from threading import RLock
from typing import Callable
from uuid import uuid4

from ai_retrieval.bulk.models import (
    ArtifactKind, BrokerDelivery, CheckpointRecord, CheckpointStage, ClaimStatus,
    DeadLetterEntry, ObjectReference, OutboxRecord, OutboxStatus,
    PersistenceFailureOutcome, ResumeState, TerminalWorkItemRecord,
    TerminalWorkItemState, WorkClaim, WorkItemRecord, WorkNotification, WorkState,
)
from ai_retrieval.bulk.effect_models import EffectRecord, ReconciliationRecord


class DurableWorkError(RuntimeError):
    pass


class WorkNotFoundError(DurableWorkError):
    pass


class WorkConflictError(DurableWorkError):
    pass


class LeaseOwnershipError(DurableWorkError):
    pass


class CheckpointTransitionError(DurableWorkError):
    pass


class InMemoryDurableWorkRepository:
    """One lock is the transaction boundary for item, checkpoint, and outbox state."""

    def __init__(self) -> None:
        self._items: dict[tuple[str, str], WorkItemRecord] = {}
        self._item_ids: dict[tuple[str, str], str] = {}
        self._checkpoints: dict[tuple[str, str], list[CheckpointRecord]] = {}
        self._terminal: dict[tuple[str, str], TerminalWorkItemRecord] = {}
        self._persistence_failures: list[PersistenceFailureOutcome] = []
        self._outbox: dict[str, OutboxRecord] = {}
        self._effects: dict[tuple[str, str], EffectRecord] = {}
        self._reconciliations: dict[tuple[str, str], ReconciliationRecord] = {}
        self._lock = RLock()

    def create_with_outbox(self, item: WorkItemRecord, outbox: OutboxRecord) -> WorkItemRecord:
        key = (item.job_id, item.idempotency_key)
        item_identity = (item.job_id, item.item_id)
        with self._lock:
            existing = self._items.get(key)
            if existing is not None:
                if existing.item_id != item.item_id or existing.configuration != item.configuration or existing.payload_reference != item.payload_reference:
                    raise WorkConflictError("idempotency key is already bound to different immutable work")
                return existing
            if item_identity in self._item_ids:
                raise WorkConflictError("work item identifier is already bound to another idempotency key")
            if outbox.event_id in self._outbox:
                raise WorkConflictError("outbox event identifier already exists")
            self._items[key] = item
            self._item_ids[item_identity] = item.idempotency_key
            self._outbox[outbox.event_id] = outbox
            return item

    def item(self, job_id: str, idempotency_key: str) -> WorkItemRecord | None:
        with self._lock:
            return self._items.get((job_id, idempotency_key))

    def items(self, job_id: str) -> tuple[WorkItemRecord, ...]:
        with self._lock:
            return tuple(sorted(
                (item for item in self._items.values() if item.job_id == job_id),
                key=lambda item: item.item_id,
            ))

    def checkpoints(self, job_id: str, idempotency_key: str) -> tuple[CheckpointRecord, ...]:
        """Return the immutable checkpoint history for recovery inspection."""
        with self._lock:
            return tuple(self._checkpoints.get((job_id, idempotency_key), ()))

    def claim(self, job_id: str, idempotency_key: str, owner: str, now: datetime, lease_duration: timedelta) -> WorkClaim:
        self._validate_lease(owner, now, lease_duration)
        key = (job_id, idempotency_key)
        with self._lock:
            current = self._require_item(key)
            latest = self._latest(key)
            if current.state is WorkState.COMPLETED:
                return WorkClaim(ClaimStatus.ACTIVE_DUPLICATE, current, latest)
            if current.state is WorkState.IN_PROGRESS and current.lease_expires_at > now:
                return WorkClaim(ClaimStatus.ACTIVE_DUPLICATE, current, latest)
            claimed = replace(
                current, state=WorkState.IN_PROGRESS, attempt_count=current.attempt_count + 1,
                lease_owner=owner, lease_expires_at=now + lease_duration, revision=current.revision + 1,
            )
            self._items[key] = claimed
            return WorkClaim(ClaimStatus.ACQUIRED, claimed, latest)

    def renew(self, job_id: str, idempotency_key: str, owner: str, now: datetime, lease_duration: timedelta) -> WorkItemRecord:
        self._validate_lease(owner, now, lease_duration)
        key = (job_id, idempotency_key)
        with self._lock:
            current = self._require_owned(key, owner, now)
            renewed = replace(current, lease_expires_at=now + lease_duration, revision=current.revision + 1)
            self._items[key] = renewed
            return renewed

    def checkpoint(
        self, job_id: str, idempotency_key: str, owner: str, stage: CheckpointStage,
        outcome: str, now: datetime, result_reference: ObjectReference | None = None,
    ) -> CheckpointRecord:
        key = (job_id, idempotency_key)
        with self._lock:
            current = self._require_owned(key, owner, now)
            latest = self._latest(key)
            if latest is not None and stage.order < latest.completed_stage.order:
                raise CheckpointTransitionError("checkpoint stages cannot move backwards")
            if latest is not None and stage == latest.completed_stage:
                if latest.outcome != outcome or latest.result_reference != result_reference:
                    raise CheckpointTransitionError("completed checkpoint stage is immutable")
                return latest
            sequence = 1 if latest is None else latest.sequence + 1
            checkpoint = CheckpointRecord(
                current.job_id, current.item_id, current.idempotency_key, stage,
                current.attempt_count, outcome, current.configuration, result_reference, now, sequence,
            )
            terminal = stage is CheckpointStage.COMPLETED
            updated = replace(
                current, state=WorkState.COMPLETED if terminal else WorkState.CHECKPOINTED,
                lease_owner=None, lease_expires_at=None,
                result_reference=result_reference or current.result_reference,
                revision=current.revision + 1,
            )
            self._checkpoints.setdefault(key, []).append(checkpoint)
            self._items[key] = updated
            return checkpoint

    def resume_state(self, job_id: str, idempotency_key: str) -> ResumeState:
        key = (job_id, idempotency_key)
        with self._lock:
            item = self._require_item(key)
            latest = self._latest(key)
            next_stage = self._next_stage(latest.completed_stage) if latest is not None else CheckpointStage.ACCEPTED
            return ResumeState(item, latest, next_stage)

    def pending_outbox(self) -> tuple[OutboxRecord, ...]:
        with self._lock:
            return tuple(record for record in self._outbox.values() if record.status is OutboxStatus.PENDING)

    def mark_published(self, event_id: str, at: datetime) -> OutboxRecord:
        with self._lock:
            record = self._outbox.get(event_id)
            if record is None:
                raise WorkNotFoundError(f"unknown outbox event: {event_id}")
            if record.status is OutboxStatus.PUBLISHED:
                return record
            published = replace(record, status=OutboxStatus.PUBLISHED, published_at=at)
            self._outbox[event_id] = published
            return published

    def terminalize(
        self, job_id: str, idempotency_key: str, owner: str,
        state: TerminalWorkItemState, failure_code: str | None,
        failure_details: str | None, now: datetime,
        result_reference: ObjectReference | None = None,
    ) -> TerminalWorkItemRecord:
        key = (job_id, idempotency_key)
        with self._lock:
            existing = self._terminal.get(key)
            if existing is not None:
                proposed = (state, failure_code, failure_details, result_reference or existing.result_reference)
                actual = (existing.state, existing.failure_code, existing.failure_details, existing.result_reference)
                if proposed != actual:
                    raise WorkConflictError("terminal work item outcome is immutable")
                return existing
            current = self._require_owned(key, owner, now)
            terminal = TerminalWorkItemRecord(
                current.job_id, current.item_id, current.idempotency_key, state,
                failure_code, failure_details, current.configuration, current.attempt_count,
                result_reference or current.result_reference, now,
            )
            self._terminal[key] = terminal
            self._items[key] = replace(
                current, state=WorkState.COMPLETED, lease_owner=None, lease_expires_at=None,
                result_reference=terminal.result_reference, revision=current.revision + 1,
            )
            return terminal

    def terminal_items(self, job_id: str) -> tuple[TerminalWorkItemRecord, ...]:
        with self._lock:
            return tuple(sorted(
                (item for item in self._terminal.values() if item.job_id == job_id),
                key=lambda item: item.item_id,
            ))

    def record_persistence_failure(self, outcome: PersistenceFailureOutcome) -> None:
        with self._lock:
            if outcome not in self._persistence_failures:
                self._persistence_failures.append(outcome)

    def persistence_failures(self, job_id: str) -> tuple[PersistenceFailureOutcome, ...]:
        with self._lock:
            return tuple(outcome for outcome in self._persistence_failures if outcome.job_id == job_id)

    def effect(self, job_id: str, idempotency_key: str) -> EffectRecord | None:
        with self._lock:
            return self._effects.get((job_id, idempotency_key))

    def save_effect(self, record: EffectRecord) -> EffectRecord:
        key = (record.job_id, record.idempotency_key)
        with self._lock:
            item = self._require_item(key)
            if item.item_id != record.item_id:
                raise WorkConflictError("effect record item does not match durable work")
            existing = self._effects.get(key)
            if existing is not None:
                if existing != record:
                    raise WorkConflictError("idempotency key is already bound to another effect record")
                return existing
            self._effects[key] = record
            return record

    def save_reconciliation(self, record: ReconciliationRecord) -> ReconciliationRecord:
        key = (record.job_id, record.idempotency_key)
        with self._lock:
            item = self._require_item(key)
            if item.item_id != record.item_id:
                raise WorkConflictError("reconciliation item does not match durable work")
            existing = self._reconciliations.get(key)
            if existing is not None:
                if (
                    existing.target_dataset != record.target_dataset
                    or existing.row_scope_digest != record.row_scope_digest
                    or existing.mutation_digest != record.mutation_digest
                    or existing.reason != record.reason
                ):
                    raise WorkConflictError("idempotency key is already awaiting different reconciliation")
                return existing
            self._reconciliations[key] = record
            return record

    def reconciliation(self, job_id: str, idempotency_key: str) -> ReconciliationRecord | None:
        with self._lock:
            return self._reconciliations.get((job_id, idempotency_key))

    def _require_item(self, key: tuple[str, str]) -> WorkItemRecord:
        item = self._items.get(key)
        if item is None:
            raise WorkNotFoundError(f"unknown work item: {key[0]}/{key[1]}")
        return item

    def _require_owned(self, key: tuple[str, str], owner: str, now: datetime) -> WorkItemRecord:
        item = self._require_item(key)
        if item.state is not WorkState.IN_PROGRESS or item.lease_owner != owner or item.lease_expires_at <= now:
            raise LeaseOwnershipError("an unexpired lease owned by the caller is required")
        return item

    def _latest(self, key: tuple[str, str]) -> CheckpointRecord | None:
        checkpoints = self._checkpoints.get(key, ())
        return checkpoints[-1] if checkpoints else None

    @staticmethod
    def _next_stage(stage: CheckpointStage) -> CheckpointStage | None:
        stages = tuple(CheckpointStage)
        index = stages.index(stage) + 1
        return stages[index] if index < len(stages) else None

    @staticmethod
    def _validate_lease(owner: str, now: datetime, lease_duration: timedelta) -> None:
        if not owner.strip() or now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("lease owner must be non-blank and time must be timezone-aware")
        if lease_duration <= timedelta(0):
            raise ValueError("lease duration must be positive")


class InMemoryObjectResultStore:
    """Content-addressed object storage; control records retain references only."""

    def __init__(self) -> None:
        self._objects: dict[str, bytes] = {}
        self._lock = RLock()

    def put_payload(self, data: bytes) -> ObjectReference:
        return self._put(data, ArtifactKind.PAYLOAD)

    def put_result(self, data: bytes) -> ObjectReference:
        return self._put(data, ArtifactKind.RESULT)

    def get(self, reference: ObjectReference) -> bytes:
        with self._lock:
            try:
                data = self._objects[reference.reference]
            except KeyError as error:
                raise KeyError(f"unknown object reference: {reference.reference}") from error
        if sha256(data).hexdigest() != reference.content_hash:
            raise ValueError("stored object does not match its content hash")
        return data

    def resolve(self, reference: str) -> ObjectReference:
        """Recover immutable content metadata from its content-addressed reference."""
        with self._lock:
            try:
                data = self._objects[reference]
            except KeyError as error:
                raise KeyError(f"unknown object reference: {reference}") from error
        prefix = "memory://"
        if not reference.startswith(prefix):
            raise ValueError("unsupported object reference")
        kind_name = reference[len(prefix):].split("/", 1)[0]
        kind = ArtifactKind(kind_name)
        return ObjectReference(reference, sha256(data).hexdigest(), kind, len(data))

    def _put(self, data: bytes, kind: ArtifactKind) -> ObjectReference:
        if not isinstance(data, bytes):
            raise TypeError("object data must be bytes")
        digest = sha256(data).hexdigest()
        reference = f"memory://{kind.value}/{digest}"
        with self._lock:
            self._objects.setdefault(reference, data)
        return ObjectReference(reference, digest, kind, len(data))


class InMemoryNotificationBroker:
    """At-least-once notification broker with explicit durable-transition acknowledgement."""

    def __init__(self, identifier: Callable[[], str] | None = None) -> None:
        self._identifier = identifier or (lambda: str(uuid4()))
        self._queued: list[WorkNotification] = []
        self._inflight: dict[str, WorkNotification] = {}
        self._lock = RLock()

    def publish(self, notification: WorkNotification) -> None:
        with self._lock:
            self._queued.append(notification)

    def receive(self) -> BrokerDelivery | None:
        with self._lock:
            if not self._queued:
                return None
            notification = self._queued.pop(0)
            delivery_id = self._identifier()
            if not delivery_id.strip() or delivery_id in self._inflight:
                raise ValueError("broker delivery identifiers must be unique and non-blank")
            self._inflight[delivery_id] = notification
            return BrokerDelivery(delivery_id, notification)

    def acknowledge(self, delivery_id: str) -> bool:
        with self._lock:
            return self._inflight.pop(delivery_id, None) is not None

    def redeliver_unacknowledged(self) -> int:
        with self._lock:
            notifications = tuple(self._inflight.values())
            self._inflight.clear()
            self._queued[0:0] = notifications
            return len(notifications)

    @property
    def unacknowledged_count(self) -> int:
        with self._lock:
            return len(self._inflight)


class DeadLetterPersistenceError(RuntimeError):
    pass


class InMemoryDeadLetterQueue:
    """Persistent prototype DLQ with deterministic fault injection by item identifier."""

    def __init__(self, fail_item_ids: frozenset[str] = frozenset()) -> None:
        self._fail_item_ids = fail_item_ids
        self._entries: dict[tuple[str, str], DeadLetterEntry] = {}
        self._lock = RLock()

    def persist(self, entry: DeadLetterEntry) -> DeadLetterEntry:
        if entry.item_id in self._fail_item_ids:
            raise DeadLetterPersistenceError(f"dead-letter persistence failed for {entry.item_id}")
        key = (entry.job_id, entry.idempotency_key)
        with self._lock:
            existing = self._entries.get(key)
            if existing is not None:
                if existing != entry:
                    raise WorkConflictError("dead-letter evidence is immutable")
                return existing
            self._entries[key] = entry
            return entry

    def entries(self, job_id: str) -> tuple[DeadLetterEntry, ...]:
        with self._lock:
            return tuple(sorted(
                (entry for entry in self._entries.values() if entry.job_id == job_id),
                key=lambda entry: entry.item_id,
            ))
