from dataclasses import replace
from datetime import datetime, timedelta, timezone

from ai_retrieval.bulk import (
    BulkCoordinator, EffectAttempt, EffectAttemptStatus, EffectEvidence,
    EffectEvidenceStatus, EffectRecoveryCoordinator, EffectRecoveryStatus,
    EffectRequest, InMemoryDurableWorkRepository, InMemoryNotificationBroker,
    InMemoryObjectResultStore, TransactionBoundary, WorkSubmission,
)
from ai_retrieval.bulk.effect_models import EffectRecord
from ai_retrieval.domain.configuration import ConfigurationReference, ExecutionConfiguration
from ai_retrieval.domain.execution import CancellationContext, DeadlineContext, ExecutionContext, ExecutionPath
from ai_retrieval.domain.identifiers import CorrelationId, ExecutionId
from ai_retrieval.domain.immutable import FrozenMapping
from ai_retrieval.relational.models import ApprovedEffect, OperationKind, OperationNode


NOW = datetime(2025, 1, 1, tzinfo=timezone.utc)
LEASE = timedelta(minutes=5)


def context():
    configuration = ExecutionConfiguration(ConfigurationReference("default", "v1"), FrozenMapping(()))
    return ExecutionContext(
        ExecutionId("execution-1"), CorrelationId("correlation-1"), ExecutionPath.BULK,
        configuration, DeadlineContext(NOW, None), CancellationContext("cancel-1", 0),
    )


def claimed_system(adapter):
    ids = iter(f"id-{index}" for index in range(10))
    repository = InMemoryDurableWorkRepository()
    broker = InMemoryNotificationBroker(identifier=lambda: next(ids))
    coordinator = BulkCoordinator(repository, InMemoryObjectResultStore(), broker, lambda: NOW, lambda: next(ids))
    coordinator.submit(WorkSubmission("job-1", "item-1", "stable-key", b"payload"), context())
    coordinator.relay_pending()
    claim = coordinator.claim_delivery(broker.receive(), "worker", LEASE)
    recovery = EffectRecoveryCoordinator(repository, repository, adapter, lambda: NOW)
    request = EffectRequest(
        claim, ApprovedEffect("stable-key", OperationNode(OperationKind.UPDATE), (("value", 1),)),
        "target", "scope-digest", "mutation-digest", adapter.boundary,
    )
    return repository, recovery, request


class RecordingAdapter:
    def __init__(self, boundary, shared=None, verification=None, non_shared=None):
        self.boundary = boundary
        self.shared = shared
        self.verification = verification or EffectEvidence(EffectEvidenceStatus.ABSENT)
        self.non_shared = non_shared
        self.calls = []

    def execute_shared(self, request, execution_context):
        self.calls.append("shared")
        if self.shared is not None:
            return self.shared
        item = request.claim.item
        return EffectEvidence(EffectEvidenceStatus.COMMITTED, EffectRecord(
            item.job_id, item.item_id, item.idempotency_key, request.target_dataset,
            request.row_scope_digest, request.mutation_digest, 1, "committed", "adapter-1", NOW,
        ))

    def verify_effect(self, idempotency_key, mutation_digest, execution_context):
        self.calls.append("verify")
        return self.verification

    def execute_non_shared(self, effect, mutation_digest, execution_context):
        self.calls.append("mutate")
        return self.non_shared or EffectAttempt(EffectAttemptStatus.COMMITTED, 1, "committed", "adapter-1")


def test_shared_transaction_atomically_records_effect_and_restores_checkpoint_without_redelivery_mutation():
    adapter = RecordingAdapter(TransactionBoundary.SHARED)
    repository, recovery, request = claimed_system(adapter)

    first = recovery.execute(request, context())
    repeated = recovery.execute(request, context())

    assert first.status is EffectRecoveryStatus.COMMITTED
    assert repeated.status is EffectRecoveryStatus.RECOVERED
    assert adapter.calls == ["shared"]
    assert repository.effect("job-1", "stable-key") == first.effect_record
    assert repository.resume_state("job-1", "stable-key").next_stage is None


def test_committed_non_shared_verification_restores_missing_checkpoint_before_any_retry():
    seed = RecordingAdapter(TransactionBoundary.NON_SHARED)
    repository, _, request = claimed_system(seed)
    item = request.claim.item
    committed = EffectRecord(
        item.job_id, item.item_id, item.idempotency_key, request.target_dataset,
        request.row_scope_digest, request.mutation_digest, 2, "committed", "adapter-2", NOW,
    )
    adapter = RecordingAdapter(
        TransactionBoundary.NON_SHARED,
        verification=EffectEvidence(EffectEvidenceStatus.COMMITTED, committed),
    )
    recovery = EffectRecoveryCoordinator(repository, repository, adapter, lambda: NOW)

    result = recovery.execute(request, context())

    assert result.status is EffectRecoveryStatus.RECOVERED
    assert adapter.calls == ["verify"]
    assert result.checkpoint.completed_stage.value == "completed"
    assert repository.effect("job-1", "stable-key") == committed


def test_non_shared_absence_verifies_before_mutation_and_persists_one_logical_outcome():
    adapter = RecordingAdapter(TransactionBoundary.NON_SHARED)
    _, recovery, request = claimed_system(adapter)

    first = recovery.execute(request, context())
    repeated = recovery.execute(request, context())

    assert first.status is EffectRecoveryStatus.COMMITTED
    assert repeated.status is EffectRecoveryStatus.RECOVERED
    assert adapter.calls == ["verify", "mutate"]
    assert first.effect_record.mutation_digest == "mutation-digest"


def test_ambiguous_verification_routes_to_durable_reconciliation_without_mutation():
    adapter = RecordingAdapter(
        TransactionBoundary.NON_SHARED,
        verification=EffectEvidence(EffectEvidenceStatus.AMBIGUOUS),
    )
    repository, recovery, request = claimed_system(adapter)

    first = recovery.execute(request, context())
    repeated = recovery.execute(request, context())

    assert first.status is EffectRecoveryStatus.RECONCILIATION_REQUIRED
    assert repeated.reconciliation == first.reconciliation
    assert adapter.calls == ["verify"]
    assert repository.effect("job-1", "stable-key") is None
    assert repository.reconciliation("job-1", "stable-key") == first.reconciliation


def test_precommit_failure_is_write_back_failed_with_no_effect_or_completed_checkpoint():
    adapter = RecordingAdapter(
        TransactionBoundary.SHARED,
        shared=EffectEvidence(EffectEvidenceStatus.ROLLED_BACK),
    )
    repository, recovery, request = claimed_system(adapter)

    result = recovery.execute(request, context())

    assert result.status is EffectRecoveryStatus.WRITE_BACK_FAILED
    assert result.reason_code == "precommit_transaction_rolled_back"
    assert repository.effect("job-1", "stable-key") is None
    assert repository.resume_state("job-1", "stable-key").next_stage.value == "accepted"
