"""Batch-only validated-output persistence and approved effect execution."""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256
import json
from typing import Callable

from ai_retrieval.bulk.effect_models import EffectRecoveryStatus, EffectRequest
from ai_retrieval.bulk.effects import EffectRecoveryCoordinator
from ai_retrieval.bulk.ports import ObjectResultStore
from ai_retrieval.domain.execution import ExecutionContext, ExecutionPath
from ai_retrieval.domain.immutable import FrozenMapping
from ai_retrieval.relational.models import ApprovedEffect, OperationNode
from ai_retrieval.write_back.models import (
    ApprovalRecord, AuthorizationSnapshot, WriteBackAuditEvent, WriteBackAuditOutcome,
    WriteBackCommand, WriteBackExecutionResult, WriteBackExecutionStatus, WriteBackPolicy,
    WriteBackScope,
)
from ai_retrieval.write_back.ports import WriteBackAuditSink, WriteBackAuthorizationSource


_SCOPE_FIELDS = (
    "job_id", "item_scope", "data_scope", "target_dataset", "row_scope", "columns",
    "mutation_type", "validation_rules_version", "configuration_version", "valid_from", "valid_until",
)


class NullWriteBackAuditSink:
    def append(self, event: WriteBackAuditEvent) -> None:
        return None


class BatchWriteBackExecutor:
    """The sole mutation owner; interactive components cannot depend on this interface."""

    def __init__(
        self,
        results: ObjectResultStore,
        authorization: WriteBackAuthorizationSource,
        recovery: EffectRecoveryCoordinator,
        clock: Callable[[], datetime],
        audit: WriteBackAuditSink | None = None,
    ) -> None:
        self._results = results
        self._authorization = authorization
        self._recovery = recovery
        self._clock = clock
        self._audit = audit or NullWriteBackAuditSink()

    def execute(self, command: WriteBackCommand, context: ExecutionContext) -> WriteBackExecutionResult:
        self._require_bulk_claim(command, context)
        validation = command.validation
        if not validation.outcome.accepted:
            result = WriteBackExecutionResult(
                WriteBackExecutionStatus.VALIDATION_REJECTED,
                tuple(validation.outcome.reason_codes) or ("validation_failed",),
            )
            self._emit_audit(command, context, None, result, None)
            return result
        if validation.outcome.rules_version != context.configuration.validation_rules_version:
            result = WriteBackExecutionResult(
                WriteBackExecutionStatus.VALIDATION_REJECTED,
                ("validation_version_not_bound",),
            )
            self._emit_audit(command, context, None, result, None)
            return result

        payload = _canonical_output(validation)
        output_reference = self._results.put_result(payload)
        enabled = _write_back_enabled(context)
        if enabled is False:
            result = WriteBackExecutionResult(
                WriteBackExecutionStatus.PERSISTED, ("write_back_disabled",),
                output_reference=output_reference,
            )
            self._emit_audit(command, context, None, result, None)
            return result
        if enabled is not True:
            result = WriteBackExecutionResult(
                WriteBackExecutionStatus.APPROVAL_REJECTED,
                ("write_back_setting_unavailable",), output_reference,
            )
            self._emit_audit(command, context, None, result, None)
            return result

        request = _effect_request(command, payload)
        snapshot = self._authorization.resolve(command.policy_version, command.approval_reference)
        checked_at = self._clock()
        reasons = _approval_failures(command, context, snapshot, checked_at)
        if reasons:
            result = WriteBackExecutionResult(
                WriteBackExecutionStatus.APPROVAL_REJECTED, reasons, output_reference,
            )
            self._emit_audit(command, context, snapshot, result, request)
            return result

        try:
            recovered = self._recovery.execute(request, context)
        except Exception as error:
            result = WriteBackExecutionResult(
                WriteBackExecutionStatus.WRITE_BACK_FAILED,
                (f"effect_execution_failure:{type(error).__name__}",), output_reference,
            )
            self._emit_audit(command, context, snapshot, result, request)
            return result
        status = {
            EffectRecoveryStatus.COMMITTED: WriteBackExecutionStatus.COMMITTED,
            EffectRecoveryStatus.RECOVERED: WriteBackExecutionStatus.RECOVERED,
            EffectRecoveryStatus.RECONCILIATION_REQUIRED: WriteBackExecutionStatus.RECONCILIATION_REQUIRED,
            EffectRecoveryStatus.WRITE_BACK_FAILED: WriteBackExecutionStatus.WRITE_BACK_FAILED,
        }[recovered.status]
        result = WriteBackExecutionResult(
            status, (recovered.reason_code,), output_reference, recovered,
        )
        self._emit_audit(command, context, snapshot, result, request)
        return result

    def _emit_audit(
        self,
        command: WriteBackCommand,
        context: ExecutionContext,
        snapshot: AuthorizationSnapshot | None,
        result: WriteBackExecutionResult,
        request: EffectRequest | None,
    ) -> None:
        outcome = {
            WriteBackExecutionStatus.COMMITTED: WriteBackAuditOutcome.COMMIT,
            WriteBackExecutionStatus.RECOVERED: WriteBackAuditOutcome.RECOVERY_DUPLICATE,
            WriteBackExecutionStatus.RECONCILIATION_REQUIRED: WriteBackAuditOutcome.RECONCILIATION,
            WriteBackExecutionStatus.WRITE_BACK_FAILED: (
                WriteBackAuditOutcome.ROLLBACK
                if "precommit_transaction_rolled_back" in result.reason_codes
                else WriteBackAuditOutcome.FAILURE
            ),
            WriteBackExecutionStatus.VALIDATION_REJECTED: WriteBackAuditOutcome.WITHHELD,
            WriteBackExecutionStatus.PERSISTED: WriteBackAuditOutcome.WITHHELD,
            WriteBackExecutionStatus.APPROVAL_REJECTED: WriteBackAuditOutcome.WITHHELD,
        }[result.status]
        approval = snapshot.approval if snapshot is not None else None
        event = WriteBackAuditEvent(
            correlation_id=str(context.correlation_id),
            job_id=command.claim.item.job_id,
            item_id=command.claim.item.item_id,
            idempotency_key=command.claim.item.idempotency_key,
            approval_reference=command.approval_reference,
            approval_authority=approval.authority if approval is not None else "unavailable",
            policy_version=command.policy_version,
            target_dataset=command.target_dataset,
            row_scope_digest=(
                request.row_scope_digest if request is not None
                else sha256(command.row_scope.encode("utf-8")).hexdigest()
            ),
            effect_digest=request.mutation_digest if request is not None else "unavailable",
            columns=tuple(sorted(command.columns)),
            mutation_type=command.mutation_type.value,
            validation_rules_version=command.validation.outcome.rules_version,
            configuration_version=context.configuration.reference.version,
            outcome=outcome,
            timestamp=self._clock(),
            reason_codes=result.reason_codes or (result.status.value,),
        )
        try:
            self._audit.append(event)
        except Exception:
            # Audit availability is intentionally outside the database transaction boundary.
            # The durable mutation/recovery outcome must never be changed by sink failure.
            pass

    @staticmethod
    def _require_bulk_claim(command: WriteBackCommand, context: ExecutionContext) -> None:
        if context.path is not ExecutionPath.BULK:
            raise ValueError("write-back executor requires a bulk execution context")
        item = command.claim.item
        if item.configuration != context.configuration.reference:
            raise ValueError("write-back context must use the work item's bound configuration")


def _write_back_enabled(context: ExecutionContext) -> bool | None:
    section = context.configuration.content.get("write_back")
    return section.get("enabled") if isinstance(section, FrozenMapping) else None


def _canonical_output(validation) -> bytes:
    records = tuple(_plain(record) for record in validation.accepted_records)
    return json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _plain(value):
    if isinstance(value, FrozenMapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, frozenset):
        return sorted((_plain(item) for item in value), key=repr)
    return value


def _effect_request(command: WriteBackCommand, payload: bytes) -> EffectRequest:
    operation = OperationNode(command.mutation_type, dataset=command.target_dataset)
    effect = ApprovedEffect(command.claim.item.idempotency_key, operation, command.parameters)
    return EffectRequest(
        command.claim, effect, command.target_dataset,
        sha256(command.row_scope.encode("utf-8")).hexdigest(),
        _mutation_digest(command, payload), command.boundary,
    )


def _mutation_digest(command: WriteBackCommand, payload: bytes) -> str:
    canonical = {
        "data_scope": command.data_scope,
        "target_dataset": command.target_dataset,
        "row_scope": command.row_scope,
        "columns": sorted(command.columns),
        "mutation_type": command.mutation_type.value,
        "parameters": [(name, repr(value)) for name, value in command.parameters],
        "validated_output_hash": sha256(payload).hexdigest(),
    }
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def _approval_failures(
    command: WriteBackCommand,
    context: ExecutionContext,
    snapshot: AuthorizationSnapshot,
    now: datetime,
) -> tuple[str, ...]:
    policy = snapshot.policy
    approval = snapshot.approval
    if policy is None:
        return ("policy_missing",)
    if approval is None:
        return ("approval_missing",)
    failures: list[str] = []
    if not policy.active:
        failures.append("policy_inactive")
    if approval.revoked:
        failures.append("approval_revoked")
    if approval.policy_version != policy.version or policy.version != command.policy_version:
        failures.append("policy_version_mismatch")
    proposed = _proposed_scope(command, context, policy.scope.valid_from, policy.scope.valid_until)
    failures.extend(
        f"policy_{field}_mismatch"
        for field in _SCOPE_FIELDS
        if getattr(policy.scope, field) != getattr(proposed, field)
    )
    failures.extend(
        f"approval_{field}_mismatch"
        for field in _SCOPE_FIELDS
        if getattr(approval.scope, field) != getattr(proposed, field)
    )
    if not (proposed.valid_from <= now < proposed.valid_until):
        failures.append("approval_outside_validity_interval")
    return tuple(dict.fromkeys(failures))


def _proposed_scope(
    command: WriteBackCommand,
    context: ExecutionContext,
    valid_from: datetime,
    valid_until: datetime,
) -> WriteBackScope:
    item = command.claim.item
    return WriteBackScope(
        item.job_id, frozenset((item.item_id,)), command.data_scope,
        command.target_dataset, command.row_scope, command.columns,
        command.mutation_type, command.validation.outcome.rules_version,
        context.configuration.reference.version, valid_from, valid_until,
    )
