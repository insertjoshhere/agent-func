"""Isolated work-item execution, DLQ handling, and terminal job reporting."""

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Iterable

from ai_retrieval.bulk.models import (
    BulkStateTelemetry, DeadLetterEntry, JobTerminalCause,
    PersistenceFailureOutcome, TerminalJobClassification, TerminalJobReport,
    TerminalStateGroup, TerminalWorkItemRecord, TerminalWorkItemState,
    WorkClaim, WorkItemExecutionResult,
)
from ai_retrieval.bulk.ports import (
    BulkTelemetrySink, DeadLetterQueue, DurableWorkRepository,
    ObjectResultStore, WorkItemHandler,
)


def classify_terminal_job(
    states: Iterable[TerminalWorkItemState], cause: JobTerminalCause,
) -> TerminalJobClassification:
    states = tuple(states)
    if not states:
        raise ValueError("a terminal job must contain at least one work item")
    succeeded = sum(state is TerminalWorkItemState.SUCCEEDED for state in states)
    if succeeded == len(states):
        return TerminalJobClassification.SUCCEEDED
    if succeeded:
        return TerminalJobClassification.PARTIALLY_SUCCEEDED
    if cause is JobTerminalCause.BUDGET_EXHAUSTED:
        return TerminalJobClassification.BUDGET_EXHAUSTED
    if cause is JobTerminalCause.CANCELLED:
        return TerminalJobClassification.CANCELLED
    return TerminalJobClassification.FAILED


def build_terminal_report(
    job_id: str, items: Iterable[TerminalWorkItemRecord],
    cause: JobTerminalCause, completed_at: datetime,
) -> TerminalJobReport:
    items = tuple(items)
    if not items:
        raise ValueError("a terminal job must contain at least one work item")
    if any(item.job_id != job_id for item in items):
        raise ValueError("terminal report items must belong to one job")
    if len({item.item_id for item in items}) != len(items):
        raise ValueError("terminal report item identifiers must be unique")
    groups = tuple(
        TerminalStateGroup(
            state,
            len(identifiers := tuple(sorted(item.item_id for item in items if item.state is state))),
            identifiers,
        )
        for state in TerminalWorkItemState
    )
    return TerminalJobReport(
        job_id, classify_terminal_job((item.state for item in items), cause),
        groups, len(items), cause, completed_at,
    )


class NullBulkTelemetrySink:
    def job_state_changed(self, telemetry: BulkStateTelemetry) -> None:
        return None


@dataclass
class _Usage:
    tokens: int = 0
    cost: int = 0


class BulkWorkExecutor:
    """Executes claims independently and persists terminal state before DLQ evidence."""

    def __init__(
        self, repository: DurableWorkRepository, objects: ObjectResultStore,
        dead_letters: DeadLetterQueue, telemetry: BulkTelemetrySink,
        clock: Callable[[], datetime], started_at: datetime,
    ) -> None:
        self._repository = repository
        self._objects = objects
        self._dead_letters = dead_letters
        self._telemetry = telemetry
        self._clock = clock
        self._started_at = started_at
        self._usage: dict[str, _Usage] = {}

    def execute_claim(self, claim: WorkClaim, handler: WorkItemHandler) -> TerminalWorkItemRecord:
        if not claim.acquired or claim.item.lease_owner is None:
            raise ValueError("an acquired claim is required")
        now = self._clock()
        try:
            result = handler.execute(claim.item)
        except Exception as error:
            result = WorkItemExecutionResult(
                TerminalWorkItemState.RETRY_EXHAUSTED,
                type(error).__name__, str(error),
            )
        result_reference = self._objects.put_result(result.result) if result.result is not None else None
        terminal = self._repository.terminalize(
            claim.item.job_id, claim.item.idempotency_key, claim.item.lease_owner,
            result.state, result.failure_code, result.failure_details, now, result_reference,
        )
        usage = self._usage.setdefault(claim.item.job_id, _Usage())
        usage.tokens += result.token_usage
        usage.cost += result.monetary_cost_minor_units
        if terminal.state is TerminalWorkItemState.RETRY_EXHAUSTED:
            self._persist_dead_letter(terminal)
        self._emit_state(claim.item.job_id)
        return terminal

    def terminal_report(self, job_id: str, cause: JobTerminalCause) -> TerminalJobReport:
        durable_items = self._repository.items(job_id)
        terminal_items = self._repository.terminal_items(job_id)
        if not durable_items:
            raise ValueError("unknown or empty bulk job")
        if len(terminal_items) != len(durable_items):
            raise ValueError("terminal report requires every work item to be terminal")
        report = build_terminal_report(job_id, terminal_items, cause, self._clock())
        self._emit_state(job_id)
        return report

    def _persist_dead_letter(self, terminal: TerminalWorkItemRecord) -> None:
        entry = DeadLetterEntry(
            terminal.job_id, terminal.item_id, terminal.idempotency_key,
            terminal.failure_code or "retry_exhausted", terminal.failure_details,
            terminal.attempt_count, terminal.configuration, terminal.result_reference,
            terminal.recorded_at,
        )
        try:
            self._dead_letters.persist(entry)
        except Exception as error:
            self._repository.record_persistence_failure(PersistenceFailureOutcome(
                terminal.job_id, terminal.item_id, "dead_letter_persist",
                f"{type(error).__name__}: {error}", self._clock(),
            ))

    def _emit_state(self, job_id: str) -> None:
        items = self._repository.terminal_items(job_id)
        counts = tuple((state, sum(item.state is state for item in items)) for state in TerminalWorkItemState)
        usage = self._usage.get(job_id, _Usage())
        now = self._clock()
        self._telemetry.job_state_changed(BulkStateTelemetry(
            job_id, counts,
            sum(max(item.attempt_count - 1, 0) for item in items),
            sum(item.state is TerminalWorkItemState.VALIDATION_FAILED for item in items),
            len(self._dead_letters.entries(job_id)), usage.tokens, usage.cost,
            max(int((now - self._started_at).total_seconds() * 1000), 0),
            len(self._repository.persistence_failures(job_id)),
        ))
