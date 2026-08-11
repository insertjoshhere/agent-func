"""Effect-record execution and recovery across transaction boundaries."""

from datetime import datetime
from typing import Callable

from ai_retrieval.bulk.effect_models import (
    EffectAttemptStatus, EffectEvidenceStatus, EffectRecord, EffectRecoveryResult,
    EffectRecoveryStatus, EffectRequest, ReconciliationRecord, TransactionBoundary,
)
from ai_retrieval.bulk.models import CheckpointStage
from ai_retrieval.bulk.ports import DurableWorkRepository, EffectAdapter, EffectRecoveryRepository
from ai_retrieval.domain.execution import ExecutionContext, ExecutionPath


class EffectRecoveryError(RuntimeError):
    pass


class EffectConflictError(EffectRecoveryError):
    pass


class EffectRecoveryCoordinator:
    """Preserves one logical mutation outcome; ambiguous effects are never replayed."""

    def __init__(
        self,
        work_repository: DurableWorkRepository,
        effect_repository: EffectRecoveryRepository,
        adapter: EffectAdapter,
        clock: Callable[[], datetime],
    ) -> None:
        self._work = work_repository
        self._effects = effect_repository
        self._adapter = adapter
        self._clock = clock

    def execute(self, request: EffectRequest, context: ExecutionContext) -> EffectRecoveryResult:
        if context.path is not ExecutionPath.BULK:
            raise ValueError("write-back effects require a bulk execution context")
        item = request.claim.item
        if item.configuration != context.configuration.reference:
            raise ValueError("effect context must use the work item's bound configuration")

        existing = self._effects.effect(item.job_id, item.idempotency_key)
        if existing is not None:
            self._assert_same_effect(existing, request)
            checkpoint = self._restore_checkpoint(request, existing)
            return EffectRecoveryResult(
                EffectRecoveryStatus.RECOVERED, existing, checkpoint,
                reason_code="committed_effect_recovered",
            )
        pending = self._effects.reconciliation(item.job_id, item.idempotency_key)
        if pending is not None:
            self._assert_same_reconciliation(pending, request)
            return EffectRecoveryResult(
                EffectRecoveryStatus.RECONCILIATION_REQUIRED,
                reconciliation=pending,
                reason_code=pending.reason,
            )

        if request.boundary is TransactionBoundary.SHARED:
            evidence = self._adapter.execute_shared(request, context)
            if evidence.status is EffectEvidenceStatus.COMMITTED:
                self._assert_same_effect(evidence.record, request)
                record = self._effects.save_effect(evidence.record)
                checkpoint = self._restore_checkpoint(request, record)
                return EffectRecoveryResult(
                    EffectRecoveryStatus.COMMITTED, record, checkpoint,
                    reason_code="transaction_committed",
                )
            if evidence.status is EffectEvidenceStatus.AMBIGUOUS:
                return self._reconcile(request, "shared_transaction_outcome_ambiguous")
            return EffectRecoveryResult(
                EffectRecoveryStatus.WRITE_BACK_FAILED,
                reason_code="precommit_transaction_rolled_back",
            )
        else:
            evidence = self._adapter.verify_effect(
                item.idempotency_key, request.mutation_digest, context,
            )
            if evidence.status is EffectEvidenceStatus.COMMITTED:
                self._assert_same_effect(evidence.record, request)
                record = self._effects.save_effect(evidence.record)
                checkpoint = self._restore_checkpoint(request, record)
                return EffectRecoveryResult(
                    EffectRecoveryStatus.RECOVERED, record, checkpoint,
                    reason_code="target_effect_recovered",
                )
            if evidence.status is EffectEvidenceStatus.AMBIGUOUS:
                return self._reconcile(request, "target_effect_verification_ambiguous")
            attempt = self._adapter.execute_non_shared(request.effect, request.mutation_digest, context)

        if attempt.status is EffectAttemptStatus.ROLLED_BACK:
            return EffectRecoveryResult(
                EffectRecoveryStatus.WRITE_BACK_FAILED,
                reason_code="precommit_transaction_rolled_back",
            )
        if attempt.status is EffectAttemptStatus.AMBIGUOUS:
            return self._reconcile(request, "mutation_commit_outcome_ambiguous")

        record = EffectRecord(
            item.job_id, item.item_id, item.idempotency_key, request.target_dataset,
            request.row_scope_digest, request.mutation_digest, attempt.affected_rows,
            attempt.transaction_outcome, attempt.adapter_version, self._clock(),
        )
        saved = self._effects.save_effect(record)
        checkpoint = self._restore_checkpoint(request, saved)
        return EffectRecoveryResult(
            EffectRecoveryStatus.COMMITTED, saved, checkpoint,
            reason_code="transaction_committed",
        )

    def _restore_checkpoint(self, request: EffectRequest, record: EffectRecord):
        latest = self._work.resume_state(record.job_id, record.idempotency_key).latest_checkpoint
        if latest is not None and latest.completed_stage is CheckpointStage.COMPLETED:
            return latest
        owner = request.claim.item.lease_owner
        if owner is None:
            raise EffectRecoveryError("committed effect recovery requires the acquired lease")
        return self._work.checkpoint(
            record.job_id, record.idempotency_key, owner, CheckpointStage.COMPLETED,
            f"write_back_committed:{record.transaction_outcome}", self._clock(),
        )

    def _reconcile(self, request: EffectRequest, reason: str) -> EffectRecoveryResult:
        item = request.claim.item
        reconciliation = self._effects.save_reconciliation(ReconciliationRecord(
            item.job_id, item.item_id, item.idempotency_key, request.target_dataset,
            request.row_scope_digest, request.mutation_digest, reason, self._clock(),
        ))
        return EffectRecoveryResult(
            EffectRecoveryStatus.RECONCILIATION_REQUIRED,
            reconciliation=reconciliation,
            reason_code=reason,
        )

    @staticmethod
    def _assert_same_reconciliation(record: ReconciliationRecord, request: EffectRequest) -> None:
        if (
            record.job_id != request.claim.item.job_id
            or record.item_id != request.claim.item.item_id
            or record.idempotency_key != request.claim.item.idempotency_key
            or record.target_dataset != request.target_dataset
            or record.row_scope_digest != request.row_scope_digest
            or record.mutation_digest != request.mutation_digest
        ):
            raise EffectConflictError("idempotency key is awaiting reconciliation for a different mutation")

    @staticmethod
    def _assert_same_effect(record: EffectRecord | None, request: EffectRequest) -> None:
        if record is None or (
            record.job_id != request.claim.item.job_id
            or record.item_id != request.claim.item.item_id
            or record.idempotency_key != request.claim.item.idempotency_key
            or record.target_dataset != request.target_dataset
            or record.row_scope_digest != request.row_scope_digest
            or record.mutation_digest != request.mutation_digest
        ):
            raise EffectConflictError("idempotency key is committed to a different mutation")
