"""Bulk admission, outbox relay, and durable notification handling."""

from datetime import datetime, timedelta
from typing import Callable
from uuid import uuid4

from ai_retrieval.bulk.models import (
    BrokerDelivery, CheckpointRecord, CheckpointStage, OutboxRecord,
    WorkClaim, WorkItemRecord, WorkNotification, WorkState, WorkSubmission,
)
from ai_retrieval.bulk.ports import DurableWorkRepository, NotificationBroker, ObjectResultStore
from ai_retrieval.domain.execution import ExecutionContext, ExecutionPath


class BulkCoordinator:
    def __init__(
        self,
        repository: DurableWorkRepository,
        objects: ObjectResultStore,
        broker: NotificationBroker,
        clock: Callable[[], datetime],
        identifier: Callable[[], str] | None = None,
    ) -> None:
        self._repository = repository
        self._objects = objects
        self._broker = broker
        self._clock = clock
        self._identifier = identifier or (lambda: str(uuid4()))

    def submit(self, submission: WorkSubmission, context: ExecutionContext) -> WorkItemRecord:
        if context.path is not ExecutionPath.BULK:
            raise ValueError("durable work may only bind to a bulk execution")
        payload_reference = self._objects.put_payload(submission.payload)
        item = WorkItemRecord(
            submission.job_id, submission.item_id, submission.idempotency_key,
            payload_reference, context.configuration.reference, WorkState.PENDING,
        )
        event_id = self._new_identifier("outbox")
        notification = WorkNotification(
            event_id, item.job_id, item.item_id, item.idempotency_key,
            item.configuration_version, payload_reference.reference,
            payload_reference.content_hash, item.revision,
        )
        return self._repository.create_with_outbox(item, OutboxRecord(event_id, notification))

    def relay_pending(self) -> int:
        published = 0
        for outbox in self._repository.pending_outbox():
            self._broker.publish(outbox.notification)
            self._repository.mark_published(outbox.event_id, self._clock())
            published += 1
        return published

    def claim_delivery(
        self, delivery: BrokerDelivery, owner: str, lease_duration: timedelta,
    ) -> WorkClaim:
        notification = delivery.notification
        item = self._repository.item(notification.job_id, notification.idempotency_key)
        if item is None:
            raise ValueError("notification references unknown work")
        if (
            item.item_id != notification.item_id
            or item.configuration_version != notification.configuration_version
            or item.payload_reference.reference != notification.payload_reference
            or item.payload_reference.content_hash != notification.payload_hash
        ):
            raise ValueError("notification metadata does not match durable work")
        claim = self._repository.claim(
            notification.job_id, notification.idempotency_key, owner, self._clock(), lease_duration,
        )
        self._broker.acknowledge(delivery.delivery_id)
        return claim

    def checkpoint(
        self, claim: WorkClaim, stage: CheckpointStage, outcome: str,
        result: bytes | None = None,
    ) -> CheckpointRecord:
        if not claim.acquired or claim.item.lease_owner is None:
            raise ValueError("an acquired claim is required")
        result_reference = self._objects.put_result(result) if result is not None else None
        return self._repository.checkpoint(
            claim.item.job_id, claim.item.idempotency_key, claim.item.lease_owner,
            stage, outcome, self._clock(), result_reference,
        )

    def _new_identifier(self, purpose: str) -> str:
        value = self._identifier()
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{purpose} identifier must not be blank")
        return value
