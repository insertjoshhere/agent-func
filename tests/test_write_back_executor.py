from dataclasses import replace
from datetime import datetime, timedelta, timezone
import inspect

import pytest

from ai_retrieval.bulk import (
    BulkCoordinator, EffectEvidence, EffectEvidenceStatus, EffectRecoveryCoordinator,
    InMemoryDurableWorkRepository, InMemoryNotificationBroker, InMemoryObjectResultStore,
    TransactionBoundary, WorkSubmission,
)
from ai_retrieval.bulk.effect_models import EffectRecord
from ai_retrieval.domain.configuration import ConfigurationReference, ExecutionConfiguration
from ai_retrieval.domain.execution import CancellationContext, DeadlineContext, ExecutionContext, ExecutionPath
from ai_retrieval.domain.identifiers import CorrelationId, ExecutionId
from ai_retrieval.domain.immutable import freeze
from ai_retrieval.interactive.coordinator import InteractiveCoordinator
from ai_retrieval.interactive.ports import ReadOnlyDataAccess
from ai_retrieval.relational.data_access import DataAccessLayer
from ai_retrieval.relational.models import OperationKind
from ai_retrieval.validation.models import ValidationOutcome, ValidationResult, ValidationStatus
from ai_retrieval.write_back import (
    ApprovalRecord, BatchWriteBackExecutor, InMemoryWriteBackAuditSink,
    InMemoryWriteBackAuthorization, WriteBackAuditOutcome, WriteBackCommand,
    WriteBackExecutionStatus, WriteBackPolicy, WriteBackScope,
)

NOW = datetime(2025, 1, 1, 12, tzinfo=timezone.utc)


def context(enabled=True, path=ExecutionPath.BULK):
    configuration = ExecutionConfiguration(
        ConfigurationReference("default", "config-1"),
        freeze({"write_back": {"enabled": enabled}}),
        "security-1", "rules-1",
    )
    return ExecutionContext(
        ExecutionId("execution-1"), CorrelationId("correlation-1"), path,
        configuration, DeadlineContext(NOW, None), CancellationContext("cancel-1", 0),
    )


def validation(accepted=True):
    status = ValidationStatus.ACCEPTED if accepted else ValidationStatus.VALIDATION_FAILED
    outcome = ValidationOutcome(
        accepted, status, () if accepted else ("domain.label",),
        () if accepted else ("value_not_allowed",), "input", "output", "rules-1",
    )
    records = (freeze({"id": "item-1", "label": "ok"}),) if accepted else ()
    return ValidationResult(outcome, records)


def scope(**changes):
    values = dict(
        job_id="job-1", item_scope=frozenset({"item-1"}), data_scope="tenant-1",
        target_dataset="results", row_scope="id=item-1", columns=frozenset({"label"}),
        mutation_type=OperationKind.UPDATE, validation_rules_version="rules-1",
        configuration_version="config-1", valid_from=NOW - timedelta(minutes=1),
        valid_until=NOW + timedelta(minutes=1),
    )
    values.update(changes)
    return WriteBackScope(**values)


class Adapter:
    boundary = TransactionBoundary.SHARED

    def __init__(self, shared=None):
        self.calls = []
        self.shared = shared

    def execute_shared(self, request, execution_context):
        self.calls.append("shared")
        if isinstance(self.shared, Exception):
            raise self.shared
        if self.shared is not None:
            return self.shared
        item = request.claim.item
        return EffectEvidence(EffectEvidenceStatus.COMMITTED, EffectRecord(
            item.job_id, item.item_id, item.idempotency_key, request.target_dataset,
            request.row_scope_digest, request.mutation_digest, 1, "committed", "adapter-1", NOW,
        ))

    def verify_effect(self, idempotency_key, mutation_digest, execution_context):
        self.calls.append("verify")
        return EffectEvidence(EffectEvidenceStatus.ABSENT)

    def execute_non_shared(self, effect, mutation_digest, execution_context):
        raise AssertionError("shared adapter must not use non-shared mutation")


def system(*, enabled=True, current_scope=None, approval=None, policy=None, audit=None, adapter=None):
    ids = iter(f"id-{index}" for index in range(10))
    repository = InMemoryDurableWorkRepository()
    objects = InMemoryObjectResultStore()
    broker = InMemoryNotificationBroker(identifier=lambda: next(ids))
    coordinator = BulkCoordinator(repository, objects, broker, lambda: NOW, lambda: next(ids))
    ctx = context(enabled)
    coordinator.submit(WorkSubmission("job-1", "item-1", "stable-key", b"payload"), ctx)
    coordinator.relay_pending()
    claim = coordinator.claim_delivery(broker.receive(), "worker", timedelta(minutes=5))
    approved_scope = current_scope or scope()
    policy = policy if policy is not None else WriteBackPolicy("policy-1", approved_scope)
    approval = approval if approval is not None else ApprovalRecord(
        "approval-1", "owner", "policy-1", approved_scope,
    )
    authorization = InMemoryWriteBackAuthorization(
        () if policy is False else (policy,), () if approval is False else (approval,),
    )
    adapter = adapter or Adapter()
    recovery = EffectRecoveryCoordinator(repository, repository, adapter, lambda: NOW)
    executor = BatchWriteBackExecutor(objects, authorization, recovery, lambda: NOW, audit)
    command = WriteBackCommand(
        claim, validation(), "policy-1", "approval-1", "tenant-1", "results",
        "id=item-1", frozenset({"label"}), OperationKind.UPDATE,
        (("label", "ok"),), TransactionBoundary.SHARED,
    )
    return ctx, objects, authorization, adapter, executor, command


def test_exact_current_approval_commits_once_and_recovers_without_repeated_mutation():
    ctx, _, _, adapter, executor, command = system()

    first = executor.execute(command, ctx)
    repeated = executor.execute(command, ctx)

    assert first.status is WriteBackExecutionStatus.COMMITTED
    assert repeated.status is WriteBackExecutionStatus.RECOVERED
    assert first.output_reference is not None
    assert adapter.calls == ["shared"]


def test_disabled_write_back_persists_validated_output_without_authorization_or_mutation():
    ctx, objects, authorization, adapter, executor, command = system(
        enabled=False, policy=False, approval=False,
    )

    result = executor.execute(command, ctx)

    assert result.status is WriteBackExecutionStatus.PERSISTED
    assert objects.get(result.output_reference) == b'[{"id":"item-1","label":"ok"}]'
    assert authorization.resolve("policy-1", "approval-1").policy is None
    assert adapter.calls == []


def test_validation_failure_is_withheld_before_persistence_authorization_or_mutation():
    ctx, _, _, adapter, executor, command = system()
    command = replace(command, validation=validation(False))

    result = executor.execute(command, ctx)

    assert result.status is WriteBackExecutionStatus.VALIDATION_REJECTED
    assert result.output_reference is None
    assert adapter.calls == []


@pytest.mark.parametrize(
    "changed_scope, reason",
    [
        (scope(job_id="job-other"), "job_id_mismatch"),
        (scope(item_scope=frozenset({"item-other"})), "item_scope_mismatch"),
        (scope(data_scope="tenant-other"), "data_scope_mismatch"),
        (scope(target_dataset="other"), "target_dataset_mismatch"),
        (scope(row_scope="id=other"), "row_scope_mismatch"),
        (scope(columns=frozenset({"other"})), "columns_mismatch"),
        (scope(mutation_type=OperationKind.DELETE), "mutation_type_mismatch"),
        (scope(validation_rules_version="rules-other"), "validation_rules_version_mismatch"),
        (scope(configuration_version="config-other"), "configuration_version_mismatch"),
        (scope(valid_from=NOW - timedelta(minutes=2)), "valid_from_mismatch"),
        (scope(valid_until=NOW + timedelta(minutes=2)), "valid_until_mismatch"),
    ],
)
def test_each_policy_scope_mismatch_rejects_before_adapter_submission(changed_scope, reason):
    ctx, _, _, adapter, executor, command = system(
        policy=WriteBackPolicy("policy-1", changed_scope),
    )

    result = executor.execute(command, ctx)

    assert result.status is WriteBackExecutionStatus.APPROVAL_REJECTED
    assert any(code.endswith(reason) for code in result.reason_codes)
    assert adapter.calls == []


@pytest.mark.parametrize(
    "policy, approval, expected",
    [
        (False, False, "policy_missing"),
        (WriteBackPolicy("policy-1", scope()), False, "approval_missing"),
        (WriteBackPolicy("policy-1", scope(), active=False), None, "policy_inactive"),
        (None, ApprovalRecord("approval-1", "owner", "policy-1", scope(), revoked=True), "approval_revoked"),
        (None, ApprovalRecord("approval-1", "owner", "other-policy", scope()), "policy_version_mismatch"),
    ],
)
def test_missing_inactive_revoked_and_unequal_authorization_reject_before_mutation(policy, approval, expected):
    ctx, _, _, adapter, executor, command = system(policy=policy, approval=approval)

    result = executor.execute(command, ctx)

    assert result.status is WriteBackExecutionStatus.APPROVAL_REJECTED
    assert expected in result.reason_codes
    assert adapter.calls == []


def test_revocation_and_expiry_are_revalidated_immediately_before_each_mutation_attempt():
    ctx, _, authorization, adapter, executor, command = system()
    authorization.revoke("approval-1")

    revoked = executor.execute(command, ctx)

    assert revoked.status is WriteBackExecutionStatus.APPROVAL_REJECTED
    assert "approval_revoked" in revoked.reason_codes
    assert adapter.calls == []

    expired_scope = scope(valid_from=NOW - timedelta(minutes=2), valid_until=NOW)
    ctx, _, _, adapter, executor, command = system(current_scope=expired_scope)
    expired = executor.execute(command, ctx)
    assert "approval_outside_validity_interval" in expired.reason_codes
    assert adapter.calls == []


def test_interactive_context_is_rejected_and_dependency_graph_has_no_mutation_contract():
    ctx, _, _, adapter, executor, command = system()
    interactive = replace(ctx, path=ExecutionPath.INTERACTIVE)

    with pytest.raises(ValueError, match="bulk execution context"):
        executor.execute(command, interactive)

    assert adapter.calls == []
    assert tuple(inspect.signature(InteractiveCoordinator.__init__).parameters) == (
        "self", "data_access", "model_planner", "model_executor", "budget", "validator",
        "telemetry", "response_reserve_ms", "clock",
    )
    assert {name for name in ReadOnlyDataAccess.__dict__ if not name.startswith("_")} == {
        "execute_read", "cancel",
    }
    assert not hasattr(ReadOnlyDataAccess, "execute_approved_effect")
    assert not hasattr(DataAccessLayer, "execute_approved_effect")


def test_commit_and_duplicate_recovery_append_complete_redacted_audit_events():
    audit = InMemoryWriteBackAuditSink()
    ctx, _, _, adapter, executor, command = system(audit=audit)

    committed = executor.execute(command, ctx)
    recovered = executor.execute(command, ctx)

    assert committed.status is WriteBackExecutionStatus.COMMITTED
    assert recovered.status is WriteBackExecutionStatus.RECOVERED
    assert tuple(event.outcome for event in audit.events) == (
        WriteBackAuditOutcome.COMMIT, WriteBackAuditOutcome.RECOVERY_DUPLICATE,
    )
    event = audit.events[0]
    assert (
        event.correlation_id, event.job_id, event.item_id, event.idempotency_key,
        event.approval_reference, event.approval_authority, event.policy_version,
        event.target_dataset, event.columns, event.mutation_type,
        event.validation_rules_version, event.configuration_version, event.timestamp,
    ) == (
        "correlation-1", "job-1", "item-1", "stable-key", "approval-1", "owner",
        "policy-1", "results", ("label",), "update", "rules-1", "config-1", NOW,
    )
    assert len(event.row_scope_digest) == 64
    assert len(event.effect_digest) == 64
    assert event.redacted_details == "[REDACTED]"
    assert "id=item-1" not in repr(event)
    assert "label', 'ok" not in repr(event)
    assert adapter.calls == ["shared"]


def test_precommit_failure_rolls_back_without_effect_and_appends_rollback_audit():
    audit = InMemoryWriteBackAuditSink()
    adapter = Adapter(EffectEvidence(EffectEvidenceStatus.ROLLED_BACK))
    ctx, _, _, _, executor, command = system(audit=audit, adapter=adapter)

    result = executor.execute(command, ctx)

    assert result.status is WriteBackExecutionStatus.WRITE_BACK_FAILED
    assert result.reason_codes == ("precommit_transaction_rolled_back",)
    assert result.effect.effect_record is None
    assert result.effect.checkpoint is None
    assert audit.events[-1].outcome is WriteBackAuditOutcome.ROLLBACK


def test_reconciliation_failure_and_withheld_outcomes_are_audited():
    reconciliation_audit = InMemoryWriteBackAuditSink()
    ambiguous = Adapter(EffectEvidence(EffectEvidenceStatus.AMBIGUOUS))
    ctx, _, _, _, executor, command = system(audit=reconciliation_audit, adapter=ambiguous)
    result = executor.execute(command, ctx)
    assert result.status is WriteBackExecutionStatus.RECONCILIATION_REQUIRED
    assert reconciliation_audit.events[-1].outcome is WriteBackAuditOutcome.RECONCILIATION

    failure_audit = InMemoryWriteBackAuditSink()
    ctx, _, _, _, executor, command = system(
        audit=failure_audit, adapter=Adapter(RuntimeError("database unavailable")),
    )
    result = executor.execute(command, ctx)
    assert result.status is WriteBackExecutionStatus.WRITE_BACK_FAILED
    assert result.reason_codes == ("effect_execution_failure:RuntimeError",)
    assert failure_audit.events[-1].outcome is WriteBackAuditOutcome.FAILURE

    withheld_audit = InMemoryWriteBackAuditSink()
    ctx, _, _, _, executor, command = system(audit=withheld_audit)
    result = executor.execute(replace(command, validation=validation(False)), ctx)
    assert result.status is WriteBackExecutionStatus.VALIDATION_REJECTED
    assert withheld_audit.events[-1].outcome is WriteBackAuditOutcome.WITHHELD


def test_audit_sink_failure_does_not_change_committed_database_outcome():
    audit = InMemoryWriteBackAuditSink(fail=True)
    ctx, _, _, adapter, executor, command = system(audit=audit)

    result = executor.execute(command, ctx)

    assert result.status is WriteBackExecutionStatus.COMMITTED
    assert result.effect.effect_record.transaction_outcome == "committed"
    assert adapter.calls == ["shared"]
    assert audit.events == ()
