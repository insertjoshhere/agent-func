import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from ai_retrieval.domain.budget import Usage
from ai_retrieval.domain.configuration import ConfigurationReference, ExecutionConfiguration
from ai_retrieval.domain.execution import CancellationContext, DeadlineContext, ExecutionContext, ExecutionPath
from ai_retrieval.domain.identifiers import CorrelationId, ExecutionId
from ai_retrieval.domain.immutable import freeze
from ai_retrieval.domain.model_routing import ModelCandidate, ModelOperation
from ai_retrieval.domain.work import InteractiveRequest, ModelWork, QueryWork
from ai_retrieval.interactive import (
    InteractiveCoordinator, InteractiveTerminalReason, ModelInvocationResult, ModelPlan,
    PartialAggregator,
)


class DataAccess:
    def __init__(self, delay=0):
        self.delay, self.contexts, self.cancelled = delay, [], []

    async def execute_read(self, reference, parameters, context):
        self.contexts.append(context)
        if self.delay:
            await asyncio.sleep(self.delay)
        return {"rows": (("1",),)}

    async def cancel(self, token):
        self.cancelled.append(token)


class Planner:
    def __init__(self):
        self.contexts, self.completed = [], []

    def plan(self, operation_id, work, context, reserve):
        self.contexts.append(context)
        operation = ModelOperation(operation_id, "tenant", "request", ExecutionPath.INTERACTIVE, work, 10,
                                   deadline=context.timing.deadline, completion_reserve_ms=reserve)
        candidate = ModelCandidate("model", "provider", work.required_capabilities, frozenset(), 2, 1, 5, 1.0)
        return ModelPlan(operation, candidate, object(), Usage(2, 8, 2))

    def complete(self, plan):
        self.completed.append(plan.operation.operation_id)


class Executor:
    def __init__(self, delay=0):
        self.delay, self.contexts, self.cancelled = delay, [], []

    async def invoke(self, plan, source, context):
        self.contexts.append(context)
        if self.delay:
            try:
                await asyncio.sleep(self.delay)
            except asyncio.CancelledError:
                await asyncio.sleep(self.delay)
        return ModelInvocationResult(({"id": plan.operation.work.input_ids[0], "label": "ok"},), Usage(1, 3, 1))

    async def cancel(self, token):
        self.cancelled.append(token)


class Budget:
    def __init__(self, exhausted=()):
        self.exhausted, self.reconciled = tuple(exhausted), []

    def reserve(self, context, estimate):
        if self.exhausted:
            return SimpleNamespace(accepted=False, exhaustion=SimpleNamespace(exhausted_scopes=self.exhausted))
        return SimpleNamespace(accepted=True, reservation=SimpleNamespace(reservation_id="reservation"))

    def reconcile(self, reservation_id, actual):
        self.reconciled.append((reservation_id, actual))


class Validator:
    def validate(self, input_ids, output, context):
        return SimpleNamespace(outcome=SimpleNamespace(accepted=True), accepted_records=tuple(output))


class Telemetry:
    def __init__(self):
        self.terminal_events, self.late = [], []

    async def terminal(self, response, metrics, context):
        self.terminal_events.append((response, metrics, context))

    async def late_completion(self, operation_id, context):
        self.late.append(operation_id)


def context(duration=1, cancellation=0):
    accepted = datetime.now(timezone.utc)
    configuration = ExecutionConfiguration(
        ConfigurationReference("default", "config-1"),
        freeze({"database": {"query_plan_version": "2"}}), "security-1", "rules-1",
    )
    return ExecutionContext(
        ExecutionId("execution"), CorrelationId("correlation"), ExecutionPath.INTERACTIVE,
        configuration, DeadlineContext(accepted, accepted + timedelta(seconds=duration)),
        CancellationContext("cancel-token", cancellation),
    )


def request(model_count=1):
    work = tuple(ModelWork("extract", f"payload-{index}", (str(index),), frozenset({"extract"}))
                 for index in range(model_count))
    return InteractiveRequest("request-1", QueryWork("customers", freeze({"minimum": 1})), work)


def test_complete_response_propagates_one_deadline_and_aggregates_validated_results():
    data, planner, executor, budget, telemetry = DataAccess(), Planner(), Executor(), Budget(), Telemetry()
    response = asyncio.run(InteractiveCoordinator(
        data, planner, executor, budget, Validator(), telemetry, response_reserve_ms=10
    ).execute(request(2), context()))

    assert response.complete and response.terminal_reason is InteractiveTerminalReason.COMPLETE
    assert [item.operation_id for item in response.results] == ["query:customers", "model:0:extract", "model:1:extract"]
    assert response.incompleteness == ()
    assert all(call.timing.deadline == response.deadline for call in data.contexts + planner.contexts + executor.contexts)
    assert len(budget.reconciled) == 2
    assert response.metrics.token_usage == 8 and response.metrics.cost_minor_units == 2
    assert len(telemetry.terminal_events) == 1


def test_budget_fallback_names_exact_model_and_exhausted_scope():
    response = asyncio.run(InteractiveCoordinator(
        DataAccess(), Planner(), Executor(), Budget(("tenant:a",)), Validator(), Telemetry(), response_reserve_ms=0
    ).execute(request(), context()))

    assert not response.complete and response.terminal_reason is InteractiveTerminalReason.BUDGET_EXHAUSTED
    assert tuple(item.operation_id for item in response.incompleteness) == ("model:0:extract",)
    assert response.incompleteness[0].details["exhausted_scopes"] == ("tenant:a",)
    assert tuple(item.operation_id for item in response.results) == ("query:customers",)


def test_deadline_fallback_is_emitted_once_cancels_and_suppresses_late_result():
    data, executor, telemetry = DataAccess(), Executor(delay=0.03), Telemetry()
    coordinator = InteractiveCoordinator(
        data, Planner(), executor, Budget(), Validator(), telemetry, response_reserve_ms=5
    )

    async def scenario():
        response = await coordinator.execute(request(), context(duration=0.01))
        await asyncio.sleep(0.05)
        return response

    response = asyncio.run(scenario())

    assert not response.complete and response.terminal_reason is InteractiveTerminalReason.DEADLINE_EXCEEDED
    assert tuple(item.operation_id for item in response.results) == ("query:customers",)
    assert tuple(item.operation_id for item in response.incompleteness) == ("model:0:extract",)
    assert executor.cancelled == ["cancel-token"]
    assert telemetry.late == ["model:0:extract"]
    assert len(telemetry.terminal_events) == 1


def test_partial_aggregator_reports_exact_pending_partition_in_requested_order():
    aggregator = PartialAggregator(("query", "model-a", "model-b"))
    aggregator.complete("query", "rows")
    aggregator.omit("model-a", InteractiveTerminalReason.VALIDATION_FAILED)
    aggregator.finalize_pending(InteractiveTerminalReason.DEADLINE_EXCEEDED)

    assert tuple(item.operation_id for item in aggregator.results()) == ("query",)
    assert tuple((item.operation_id, item.reason) for item in aggregator.incompleteness()) == (
        ("model-a", InteractiveTerminalReason.VALIDATION_FAILED),
        ("model-b", InteractiveTerminalReason.DEADLINE_EXCEEDED),
    )
