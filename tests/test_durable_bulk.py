from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

from ai_retrieval.bulk import (
    BulkCoordinator,
    CheckpointStage,
    CheckpointTransitionError,
    ClaimStatus,
    InMemoryDurableWorkRepository,
    InMemoryNotificationBroker,
    InMemoryObjectResultStore,
    WorkConflictError,
    WorkState,
    WorkSubmission,
)
from ai_retrieval.domain.configuration import ConfigurationReference, ExecutionConfiguration
from ai_retrieval.domain.execution import CancellationContext, DeadlineContext, ExecutionContext, ExecutionPath
from ai_retrieval.domain.identifiers import CorrelationId, ExecutionId
from ai_retrieval.domain.immutable import FrozenMapping


NOW = datetime(2025, 1, 1, tzinfo=timezone.utc)
LEASE = timedelta(minutes=5)


def context(version: str = "v1", path: ExecutionPath = ExecutionPath.BULK) -> ExecutionContext:
    configuration = ExecutionConfiguration(ConfigurationReference("default", version), FrozenMapping(()))
    return ExecutionContext(
        ExecutionId("execution-1"), CorrelationId("correlation-1"), path, configuration,
        DeadlineContext(NOW, None), CancellationContext("cancel-1", 0),
    )


def system(now=lambda: NOW):
    identifiers = iter(f"id-{index}" for index in range(1, 100))
    repository = InMemoryDurableWorkRepository()
    objects = InMemoryObjectResultStore()
    broker = InMemoryNotificationBroker(identifier=lambda: next(identifiers))
    coordinator = BulkCoordinator(repository, objects, broker, now, identifier=lambda: next(identifiers))
    return repository, objects, broker, coordinator


def submit(coordinator: BulkCoordinator, payload: bytes = b"large payload", version: str = "v1"):
    return coordinator.submit(WorkSubmission("job-1", "item-1", "stable-key", payload), context(version))


def deliver(coordinator: BulkCoordinator, broker: InMemoryNotificationBroker, owner: str = "worker-1"):
    assert coordinator.relay_pending() == 1
    delivery = broker.receive()
    assert delivery is not None
    return delivery, coordinator.claim_delivery(delivery, owner, LEASE)


def test_submission_atomically_persists_item_and_outbox_with_reference_only_notification():
    repository, objects, _, coordinator = system()

    item = submit(coordinator, b"x" * 100_000)
    outbox = repository.pending_outbox()

    assert item.state is WorkState.PENDING
    assert item.configuration_version == "v1"
    assert objects.get(item.payload_reference) == b"x" * 100_000
    assert len(outbox) == 1
    notification = outbox[0].notification
    assert notification.payload_reference == item.payload_reference.reference
    assert notification.payload_hash == item.payload_reference.content_hash
    assert not hasattr(notification, "payload")


def test_submission_requires_stable_key_and_rejects_conflicting_duplicate_binding():
    _, _, _, coordinator = system()
    with pytest.raises(ValueError):
        coordinator.submit(WorkSubmission("job-1", "item-1", " ", b"payload"), context())

    original = submit(coordinator)
    assert submit(coordinator) == original
    with pytest.raises(WorkConflictError):
        coordinator.submit(WorkSubmission("job-1", "item-1", "stable-key", b"different"), context())


def test_active_duplicate_claims_are_suppressed_atomically_under_concurrency():
    repository, _, broker, coordinator = system()
    submit(coordinator)
    assert coordinator.relay_pending() == 1
    delivery = broker.receive()
    assert delivery is not None

    with ThreadPoolExecutor(max_workers=12) as pool:
        claims = tuple(pool.map(lambda index: repository.claim("job-1", "stable-key", f"worker-{index}", NOW, LEASE), range(12)))

    assert sum(claim.status is ClaimStatus.ACQUIRED for claim in claims) == 1
    assert sum(claim.status is ClaimStatus.ACTIVE_DUPLICATE for claim in claims) == 11
    assert repository.item("job-1", "stable-key").attempt_count == 1


def test_notification_is_acknowledged_only_after_durable_claim_and_invalid_delivery_remains_unacked():
    _, _, broker, coordinator = system()
    submit(coordinator)
    assert coordinator.relay_pending() == 1
    delivery = broker.receive()

    original = delivery.notification
    tampered = type(original)(
        original.event_id, original.job_id, original.item_id, original.idempotency_key,
        original.configuration_version, original.payload_reference, "0" * 64, original.aggregate_revision,
    )
    with pytest.raises(ValueError):
        coordinator.claim_delivery(type(delivery)(delivery.delivery_id, tampered), "worker", LEASE)
    assert broker.unacknowledged_count == 1

    claim = coordinator.claim_delivery(delivery, "worker", LEASE)
    assert claim.acquired
    assert broker.unacknowledged_count == 0


def test_expired_lease_resumes_latest_checkpoint_without_changing_idempotency_or_configuration():
    mutable_now = [NOW]
    repository, objects, broker, coordinator = system(lambda: mutable_now[0])
    item = submit(coordinator)
    _, first = deliver(coordinator, broker)
    result = objects.put_result(b"model output")
    checkpoint = repository.checkpoint(
        item.job_id, item.idempotency_key, "worker-1", CheckpointStage.MODEL_COMPLETED,
        "model_succeeded", mutable_now[0], result,
    )

    mutable_now[0] += LEASE + timedelta(seconds=1)
    resumed = repository.claim(item.job_id, item.idempotency_key, "worker-2", mutable_now[0], LEASE)

    assert resumed.acquired and resumed.item.attempt_count == 2
    assert resumed.item.idempotency_key == "stable-key"
    assert resumed.item.configuration_version == "v1"
    assert resumed.latest_checkpoint == checkpoint
    assert repository.resume_state(item.job_id, item.idempotency_key).next_stage is CheckpointStage.RESULT_STORED


def test_checkpoint_is_monotonic_contains_required_recovery_fields_and_completed_stage_is_not_reclaimed():
    repository, _, broker, coordinator = system()
    item = submit(coordinator)
    _, claim = deliver(coordinator, broker)
    checkpoint = coordinator.checkpoint(claim, CheckpointStage.VALIDATED, "accepted", b"validated result")

    assert checkpoint.idempotency_key == "stable-key"
    assert checkpoint.attempt_count == 1
    assert checkpoint.outcome == "accepted"
    assert checkpoint.configuration_version == "v1"
    assert checkpoint.result_reference is not None

    resumed = repository.claim(item.job_id, item.idempotency_key, "worker-2", NOW + timedelta(seconds=1), LEASE)
    with pytest.raises(CheckpointTransitionError):
        repository.checkpoint(item.job_id, item.idempotency_key, "worker-2", CheckpointStage.MODEL_COMPLETED, "old", NOW + timedelta(seconds=2))

    completed = repository.checkpoint(item.job_id, item.idempotency_key, "worker-2", CheckpointStage.COMPLETED, "succeeded", NOW + timedelta(seconds=2))
    assert completed.completed_stage is CheckpointStage.COMPLETED
    assert repository.resume_state(item.job_id, item.idempotency_key).next_stage is None
    duplicate = repository.claim(item.job_id, item.idempotency_key, "worker-3", NOW + timedelta(days=1), LEASE)
    assert duplicate.status is ClaimStatus.ACTIVE_DUPLICATE


def test_bound_configuration_and_payload_are_immutable_across_profile_change():
    repository, objects, _, coordinator = system()
    item = submit(coordinator, b"payload-v1", version="v1")

    assert item.configuration_version == "v1"
    assert objects.get(item.payload_reference) == b"payload-v1"
    with pytest.raises(WorkConflictError):
        coordinator.submit(WorkSubmission("job-1", "item-1", "stable-key", b"payload-v1"), context("v2"))
    assert repository.item("job-1", "stable-key").configuration_version == "v1"


def test_bulk_coordinator_rejects_interactive_context_before_durable_work():
    repository, _, _, coordinator = system()
    with pytest.raises(ValueError):
        coordinator.submit(WorkSubmission("job", "item", "key", b"payload"), context(path=ExecutionPath.INTERACTIVE))
    assert repository.pending_outbox() == ()
