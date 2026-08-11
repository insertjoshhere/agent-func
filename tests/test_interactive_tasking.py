"""Focused task-aware interactive preparation and coordinator tests."""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from ai_retrieval.domain.budget import Usage
from ai_retrieval.domain.configuration import ConfigurationReference, ExecutionConfiguration
from ai_retrieval.domain.execution import CancellationContext, DeadlineContext, ExecutionContext, ExecutionPath
from ai_retrieval.domain.identifiers import CorrelationId, ExecutionId
from ai_retrieval.domain.immutable import FrozenMapping, freeze
from ai_retrieval.domain.model_routing import ModelCandidate, ModelOperation
from ai_retrieval.domain.work import InteractiveRequest, InteractiveTaskWork, ModelWork, QueryWork
from ai_retrieval.interactive import (
    ConfiguredTaskRowAdapter, InteractiveTaskProcessor,
    InteractiveTerminalReason, ModelInvocationResult, ModelPlan,
    TaskAwareInteractiveCoordinator,
)
from ai_retrieval.relational import NormalizedColumn, NormalizedResult, NormalizedType
from ai_retrieval.relational.models import ProtectionMetadata
from ai_retrieval.tasks import (
    PackedRequest, PackingLimits, PreparedTask, RowInput, TaskDefinition, TaskFailure,
    TaskFailureCode, TaskFailureStage, TaskFunction, TaskInvocation, TaskOutputFailure,
)
from ai_retrieval.validation.models import ValidationOutcome, ValidationResult, ValidationStatus


def result(rows=(("row-2", "second"), ("row-1", "first"))):
    return NormalizedResult(
        (NormalizedColumn("record_id", NormalizedType.STRING, False),
         NormalizedColumn("body", NormalizedType.STRING, False)),
        rows, None, None, "adapter-1",
        ProtectionMetadata("security-1", "reader", "tls", "none"),
    )


def context(*, row_mapping=True, duration=1):
    accepted = datetime.now(timezone.utc)
    interactive = {
        "task_input": {
            "identifier_column": "record_id",
            "source_fields": {"text": "body"},
        }
    } if row_mapping else {}
    configuration = ExecutionConfiguration(
        ConfigurationReference("default", "config-1"),
        freeze({"database": {"query_plan_version": "2"}, "interactive": interactive}),
        "security-1", "rules-1",
    )
    return ExecutionContext(
        ExecutionId("execution"), CorrelationId("correlation"), ExecutionPath.INTERACTIVE,
        configuration, DeadlineContext(accepted, accepted + timedelta(seconds=duration)),
        CancellationContext("cancel-token", 0),
    )


def task_request():
    return InteractiveRequest(
        "request-1", QueryWork("customers"), (),
        InteractiveTaskWork("ai_summarize", freeze({"max_words": 2})),
    )


class DataAccess:
    def __init__(self):
        self.cancelled = []

    async def execute_read(self, reference, parameters, execution_context):
        return result()

    async def cancel(self, token):
        self.cancelled.append(token)


class Planner:
    def __init__(self):
        self.operation_ids, self.work, self.completed = [], [], []

    def plan(self, operation_id, work, execution_context, reserve):
        self.operation_ids.append(operation_id)
        self.work.append(work)
        operation = ModelOperation(
            operation_id, "tenant", "request", ExecutionPath.INTERACTIVE, work,
            work.estimated_input_tokens + work.estimated_output_tokens,
            deadline=execution_context.timing.deadline, completion_reserve_ms=reserve,
        )
        candidate = ModelCandidate(
            "model", "provider", work.required_capabilities, frozenset(), 2, 1, 5, 1.0
        )
        return ModelPlan(operation, candidate, object(), Usage(2, work.estimated_input_tokens, work.estimated_output_tokens))

    def complete(self, plan):
        self.completed.append(plan.operation.operation_id)


class Executor:
    def __init__(self, responses):
        self.responses, self.cancelled = list(responses), []

    async def invoke(self, plan, source, execution_context):
        return ModelInvocationResult(self.responses.pop(0), Usage(1, 2, 1))

    async def cancel(self, token):
        self.cancelled.append(token)


class Budget:
    def __init__(self):
        self.estimates, self.reconciled = [], []

    def reserve(self, execution_context, estimate):
        self.estimates.append(estimate)
        return SimpleNamespace(accepted=True, reservation=SimpleNamespace(reservation_id=f"r-{len(self.estimates)}"))

    def reconcile(self, reservation_id, actual):
        self.reconciled.append((reservation_id, actual))


class Telemetry:
    def __init__(self):
        self.terminal_events = []

    async def terminal(self, response, metrics, execution_context):
        self.terminal_events.append(response)

    async def late_completion(self, operation_id, execution_context):
        pass


class PreparedRuntime:
    def __init__(self):
        self.invocation = None

    def prepare(self, invocation, execution_context):
        self.invocation = invocation
        definition = TaskDefinition(
            invocation.function, "definition-1", "Summarize", freeze({}), freeze({}), freeze({}),
            frozenset({"ai_summarize"}), "rules-1", PackingLimits(1, 20, 5, 4096), 4096,
        )
        packs = tuple(
            PackedRequest(invocation.function, definition.version, index, (row.identifier,), b"{}", 10 + index, 2)
            for index, row in enumerate(invocation.rows)
        )
        works = tuple(
            ModelWork("ai_summarize", f"payload-{index}", pack.input_ids,
                      definition.required_capabilities, definition.version,
                      pack.estimated_input_tokens, pack.estimated_output_tokens)
            for index, pack in enumerate(packs)
        )
        return PreparedTask(definition, invocation, packs, works)


class Parser:
    def parse(self, response, definition):
        if response == "bad-json":
            return TaskOutputFailure(
                TaskFailureStage.PARSING, TaskFailureCode.MALFORMED_STRUCTURED_OUTPUT,
                ("structured_output.document",), freeze({"reason": "invalid_json"}),
            )
        return tuple(freeze(record) for record in response)


class Validator:
    def validate(self, prepared, pack, output, execution_context):
        accepted = all(record["summary"] != "too long summary" for record in output)
        outcome = ValidationOutcome(
            accepted, ValidationStatus.ACCEPTED if accepted else ValidationStatus.VALIDATION_FAILED,
            () if accepted else ("task_output.summarize.summary.word_limit",),
            () if accepted else ("summary_word_limit_exceeded",), "in", "out", "rules-1",
        )
        return ValidationResult(outcome, tuple(output) if accepted else (), accepted)


def processor(runtime):
    return InteractiveTaskProcessor(runtime, Parser(), Validator())


def test_configured_row_adapter_preserves_result_order_and_selected_field_names():
    rows = ConfiguredTaskRowAdapter().rows(result(), context())

    assert not isinstance(rows, TaskFailure)
    assert tuple(row.identifier for row in rows) == ("row-2", "row-1")
    assert tuple(row.source_fields for row in rows) == (
        freeze({"text": "second"}), freeze({"text": "first"}),
    )


def test_configured_row_adapter_returns_typed_failure_for_missing_columns():
    failure = ConfiguredTaskRowAdapter().rows(result(), context(row_mapping=False))

    assert isinstance(failure, TaskFailure)
    assert failure.code is TaskFailureCode.INVALID_TASK_INVOCATION
    assert failure.failed_rule_ids == ("interactive_task.row_mapping",)
    assert failure.details["reason"] == "row_mapping_missing"


def test_task_request_rejects_mixed_legacy_model_work():
    try:
        InteractiveRequest(
            "request", QueryWork("customers"), (ModelWork("legacy", "payload", ("1",)),),
            InteractiveTaskWork("ai_summarize", freeze({"max_words": 2})),
        )
    except ValueError as error:
        assert "cannot mix" in str(error)
    else:
        raise AssertionError("mixed task and legacy work must be rejected")


def test_task_aware_coordinator_prepares_after_read_and_executes_stable_bounded_packs():
    runtime = PreparedRuntime()
    planner, budget, telemetry = Planner(), Budget(), Telemetry()
    coordinator = TaskAwareInteractiveCoordinator(
        DataAccess(), planner,
        Executor((({"id": "row-2", "summary": "second"},),
                  ({"id": "row-1", "summary": "first"},))),
        budget, SimpleNamespace(), telemetry, processor(runtime), response_reserve_ms=0,
    )

    response = asyncio.run(coordinator.execute(task_request(), context()))

    assert runtime.invocation.rows == (
        RowInput("row-2", freeze({"text": "second"})),
        RowInput("row-1", freeze({"text": "first"})),
    )
    assert planner.operation_ids == [
        "task:ai_summarize:definition-1:pack:0",
        "task:ai_summarize:definition-1:pack:1",
    ]
    assert budget.estimates == [Usage(2, 10, 2), Usage(2, 11, 2)]
    assert response.complete
    assert tuple(item.operation_id for item in response.results) == (
        "query:customers",
        "task:ai_summarize:definition-1:pack:0",
        "task:ai_summarize:definition-1:pack:1",
    )
    assert tuple(record["id"] for item in response.results[1:] for record in item.value) == ("row-2", "row-1")
    assert len(telemetry.terminal_events) == 1


def test_parser_failure_maps_exact_pack_incompleteness_without_raw_response():
    runtime = PreparedRuntime()
    response = asyncio.run(TaskAwareInteractiveCoordinator(
        DataAccess(), Planner(), Executor(("bad-json", ({"id": "row-1", "summary": "first"},))),
        Budget(), SimpleNamespace(), Telemetry(), processor(runtime), response_reserve_ms=0,
    ).execute(task_request(), context()))

    assert response.terminal_reason is InteractiveTerminalReason.VALIDATION_FAILED
    assert tuple(item.operation_id for item in response.incompleteness) == (
        "task:ai_summarize:definition-1:pack:0",
    )
    details = response.incompleteness[0].details
    assert details["failure_code"] == "malformed_structured_output"
    assert details["failed_rule_ids"] == ("structured_output.document",)
    assert "bad-json" not in repr(details)
    assert tuple(item.operation_id for item in response.results) == (
        "query:customers", "task:ai_summarize:definition-1:pack:1",
    )


def test_preparation_failure_has_exact_typed_entry_and_no_planner_or_budget_effects():
    planner, budget = Planner(), Budget()
    response = asyncio.run(TaskAwareInteractiveCoordinator(
        DataAccess(), planner, Executor(()), budget, SimpleNamespace(), Telemetry(),
        processor(PreparedRuntime()), response_reserve_ms=0,
    ).execute(task_request(), context(row_mapping=False)))

    assert response.terminal_reason is InteractiveTerminalReason.VALIDATION_FAILED
    assert tuple(item.operation_id for item in response.incompleteness) == (
        "task:ai_summarize:prepare",
    )
    assert response.incompleteness[0].details["failure_code"] == "invalid_task_invocation"
    assert planner.operation_ids == [] and budget.estimates == []
