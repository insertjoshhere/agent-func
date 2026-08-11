from datetime import datetime, timedelta, timezone

import pytest

from ai_retrieval.bulk import (
    BulkCoordinator, BulkWorkExecutor, InMemoryDeadLetterQueue,
    InMemoryDurableWorkRepository, InMemoryNotificationBroker,
    InMemoryObjectResultStore, JobTerminalCause, TerminalJobClassification,
    TerminalWorkItemState, WorkItemExecutionResult, WorkSubmission,
    classify_terminal_job,
)
from ai_retrieval.domain.configuration import ConfigurationReference, ExecutionConfiguration
from ai_retrieval.domain.execution import CancellationContext, DeadlineContext, ExecutionContext, ExecutionPath
from ai_retrieval.domain.identifiers import CorrelationId, ExecutionId
from ai_retrieval.domain.immutable import FrozenMapping

NOW = datetime(2025, 1, 1, tzinfo=timezone.utc)
LEASE = timedelta(minutes=5)


def context():
    configuration = ExecutionConfiguration(ConfigurationReference("default", "v1"), FrozenMapping(()))
    return ExecutionContext(
        ExecutionId("execution-1"), CorrelationId("correlation-1"), ExecutionPath.BULK,
        configuration, DeadlineContext(NOW, None), CancellationContext("cancel-1", 0),
    )


class Telemetry:
    def __init__(self):
        self.events = []

    def job_state_changed(self, telemetry):
        self.events.append(telemetry)


class Handler:
    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.called = []

    def execute(self, item):
        self.called.append(item.item_id)
        outcome = self.outcomes[item.item_id]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def system(fail_dlq=frozenset()):
    identifiers = iter(f"id-{index}" for index in range(100))
    repository = InMemoryDurableWorkRepository()
    objects = InMemoryObjectResultStore()
    broker = InMemoryNotificationBroker(lambda: next(identifiers))
    coordinator = BulkCoordinator(repository, objects, broker, lambda: NOW, lambda: next(identifiers))
    telemetry = Telemetry()
    dead_letters = InMemoryDeadLetterQueue(fail_dlq)
    executor = BulkWorkExecutor(repository, objects, dead_letters, telemetry, lambda: NOW, NOW)
    return repository, broker, coordinator, dead_letters, telemetry, executor


def claims(coordinator, broker, item_ids):
    for item_id in item_ids:
        coordinator.submit(WorkSubmission("job-1", item_id, f"key-{item_id}", item_id.encode()), context())
    coordinator.relay_pending()
    output = []
    for index in range(len(item_ids)):
        delivery = broker.receive()
        output.append(coordinator.claim_delivery(delivery, f"worker-{index}", LEASE))
    return output


def test_item_failure_is_isolated_and_eligible_peer_continues():
    repository, broker, coordinator, dead_letters, _, executor = system()
    item_claims = claims(coordinator, broker, ("bad", "good"))
    handler = Handler({
        "bad": RuntimeError("provider failed"),
        "good": WorkItemExecutionResult(TerminalWorkItemState.SUCCEEDED, result=b"ok"),
    })

    failed = executor.execute_claim(item_claims[0], handler)
    succeeded = executor.execute_claim(item_claims[1], handler)

    assert failed.state is TerminalWorkItemState.RETRY_EXHAUSTED
    assert succeeded.state is TerminalWorkItemState.SUCCEEDED
    assert handler.called == ["bad", "good"]
    assert tuple(entry.item_id for entry in dead_letters.entries("job-1")) == ("bad",)
    assert repository.terminal_items("job-1") == (failed, succeeded)


def test_only_retry_exhausted_items_enter_dlq_and_failure_preserves_terminal_state():
    repository, broker, coordinator, dead_letters, telemetry, executor = system(frozenset({"retry"}))
    item_claims = claims(coordinator, broker, ("validation", "retry"))
    handler = Handler({
        "validation": WorkItemExecutionResult(TerminalWorkItemState.VALIDATION_FAILED, "schema", "invalid"),
        "retry": WorkItemExecutionResult(TerminalWorkItemState.RETRY_EXHAUSTED, "timeout", "attempts exhausted"),
    })

    validation = executor.execute_claim(item_claims[0], handler)
    retry = executor.execute_claim(item_claims[1], handler)

    assert validation.state is TerminalWorkItemState.VALIDATION_FAILED
    assert retry.state is TerminalWorkItemState.RETRY_EXHAUSTED
    assert dead_letters.entries("job-1") == ()
    failure = repository.persistence_failures("job-1")
    assert len(failure) == 1 and failure[0].item_id == "retry"
    assert telemetry.events[-1].persistence_failure_count == 1


@pytest.mark.parametrize(
    ("states", "cause", "expected"),
    [
        ((TerminalWorkItemState.SUCCEEDED,), JobTerminalCause.COMPLETED, TerminalJobClassification.SUCCEEDED),
        ((TerminalWorkItemState.SUCCEEDED, TerminalWorkItemState.CANCELLED), JobTerminalCause.CANCELLED, TerminalJobClassification.PARTIALLY_SUCCEEDED),
        ((TerminalWorkItemState.BUDGET_EXHAUSTED,), JobTerminalCause.BUDGET_EXHAUSTED, TerminalJobClassification.BUDGET_EXHAUSTED),
        ((TerminalWorkItemState.CANCELLED,), JobTerminalCause.CANCELLED, TerminalJobClassification.CANCELLED),
        ((TerminalWorkItemState.POLICY_REJECTED,), JobTerminalCause.COMPLETED, TerminalJobClassification.FAILED),
    ],
)
def test_terminal_job_classification_is_total_and_deterministic(states, cause, expected):
    assert classify_terminal_job(states, cause) is expected
    assert classify_terminal_job(states, cause) is expected


def test_terminal_report_is_complete_partition_and_telemetry_has_required_counts():
    _, broker, coordinator, _, telemetry, executor = system()
    item_claims = claims(coordinator, broker, ("success", "cancelled", "retry"))
    handler = Handler({
        "success": WorkItemExecutionResult(TerminalWorkItemState.SUCCEEDED, result=b"ok", token_usage=5, monetary_cost_minor_units=2),
        "cancelled": WorkItemExecutionResult(TerminalWorkItemState.CANCELLED, "cancelled", "operator request"),
        "retry": WorkItemExecutionResult(TerminalWorkItemState.RETRY_EXHAUSTED, "timeout", "attempts exhausted"),
    })
    for claim in item_claims:
        executor.execute_claim(claim, handler)

    report = executor.terminal_report("job-1", JobTerminalCause.CANCELLED)
    grouped = {group.state: group for group in report.groups}

    assert report.classification is TerminalJobClassification.PARTIALLY_SUCCEEDED
    assert report.total_count == 3
    assert grouped[TerminalWorkItemState.SUCCEEDED].item_ids == ("success",)
    assert grouped[TerminalWorkItemState.CANCELLED].item_ids == ("cancelled",)
    assert grouped[TerminalWorkItemState.RETRY_EXHAUSTED].item_ids == ("retry",)
    assert sum(group.count for group in report.groups) == report.total_count
    latest = telemetry.events[-1]
    assert latest.dead_letter_count == 1
    assert latest.token_usage == 5 and latest.monetary_cost_minor_units == 2
    assert dict(latest.counts_by_state)[TerminalWorkItemState.RETRY_EXHAUSTED] == 1


def test_terminal_report_rejects_nonterminal_job():
    _, broker, coordinator, _, _, executor = system()
    item_claims = claims(coordinator, broker, ("done", "pending"))
    executor.execute_claim(item_claims[0], Handler({
        "done": WorkItemExecutionResult(TerminalWorkItemState.SUCCEEDED),
    }))

    with pytest.raises(ValueError, match="every work item"):
        executor.terminal_report("job-1", JobTerminalCause.COMPLETED)
