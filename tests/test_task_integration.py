"""Composed cross-path coverage for provider-neutral task execution."""

import asyncio
import json

import pytest

from ai_retrieval.bulk import CheckpointStage, TerminalWorkItemState
from ai_retrieval.composition import build_prototype
from ai_retrieval.control_plane.budget import InMemoryBudgetController
from ai_retrieval.domain.budget import BudgetLimit, Usage
from ai_retrieval.domain.immutable import FrozenMapping, freeze
from ai_retrieval.interactive import InteractiveTerminalReason, ModelInvocationResult
from ai_retrieval.tasks import TaskFunction


def _task_operation(app, function, pack_index=0):
    return f"task:{function.value}:{app.task_definition_versions[function.value]}:pack:{pack_index}"


def _events(app, event_type):
    return tuple(
        event for event in app.telemetry.events
        if isinstance(event, FrozenMapping) and event.get("event_type") == event_type
    )


@pytest.mark.parametrize(
    "function,parameters,field,expected",
    (
        (TaskFunction.CLASSIFY, freeze({"labels": ("approved", "rejected")}), "label", "approved"),
        (TaskFunction.SUMMARIZE, freeze({"max_words": 1}), "summary", "Ada"),
    ),
)
def test_interactive_task_success_preserves_routing_budget_telemetry_and_read_only_contract(
    function, parameters, field, expected,
):
    app = build_prototype()

    decision = app.admit_interactive_task(
        f"interactive-{function.value}", "customers", 1.0, function, parameters,
    )

    response = decision.outcome
    assert decision.path.value == "interactive"
    assert response.complete
    assert response.terminal_reason is InteractiveTerminalReason.COMPLETE
    assert tuple(item.operation_id for item in response.results) == (
        "query:customers", _task_operation(app, function),
    )
    assert tuple(dict(record) for record in response.results[1].value) == (
        {"id": "customer-1", field: expected},
    )
    assert response.configuration_version == app.reference.version
    assert response.metrics.token_usage == 4
    assert response.metrics.cost_minor_units == 1
    assert app.budget.balance("prototype").actual == Usage(1, 3, 1)
    assert app.read_adapter.read_count == 1
    assert app.model_provider.calls == 1
    assert app.effect_adapter.mutation_count == 0
    terminal = _events(app, "interactive_terminal")
    assert len(terminal) == 1
    assert terminal[0]["correlation_id"] == str(decision.context.correlation_id)
    assert terminal[0]["outcome"] == "complete"


@pytest.mark.parametrize(
    "parameters,expected_rule",
    (
        (freeze({"labels": ()}), "task_invocation.classify.labels.nonempty"),
        (freeze({"labels": ("duplicate", "duplicate")}), "task_invocation.classify.labels.unique"),
    ),
)
def test_interactive_parameter_failure_is_typed_before_model_budget_or_mutation(
    parameters, expected_rule,
):
    app = build_prototype()

    response = app.admit_interactive_task(
        "invalid-parameters", "customers", 1.0, TaskFunction.CLASSIFY, parameters,
    ).outcome

    failure = response.incompleteness[0]
    assert response.terminal_reason is InteractiveTerminalReason.VALIDATION_FAILED
    assert failure.operation_id == "task:ai_classify:prepare"
    assert failure.details["failure_code"] == "invalid_task_invocation"
    assert expected_rule in failure.details["failed_rule_ids"]
    assert app.model_provider.calls == 0
    assert app.budget.balance("prototype").actual == Usage()
    assert app.effect_adapter.mutation_count == 0


def test_bulk_oversized_row_fails_packing_before_model_budget_checkpoint_or_mutation():
    app = build_prototype(write_back_enabled=True)
    decision = app.admit_bulk_task(
        "bulk-job", ("item-1",), TaskFunction.SUMMARIZE,
        freeze({"max_words": 10}),
        source_rows=freeze({"item-1": {"text": "x" * 80_000}}),
    )

    result = app.run_bulk(decision)
    terminal = app.repository.terminal_items("bulk-job")[0]

    assert result.classification == "failed"
    assert terminal.state is TerminalWorkItemState.VALIDATION_FAILED
    assert terminal.failure_code == "oversized_row"
    assert "packing" in terminal.failure_details
    assert app.repository.checkpoints("bulk-job", "bulk-job:item-1") == ()
    assert app.model_provider.calls == 0
    assert app.budget.balance("prototype").actual == Usage()
    assert result.mutation_count == 0


@pytest.mark.parametrize(
    "provider_output,expected_code",
    (
        (b"{not-json", "malformed_structured_output"),
        (({"id": "item-1", "unexpected": "value"},), "output_schema_violation"),
    ),
)
def test_bulk_parser_failures_are_terminal_and_withhold_write_back(provider_output, expected_code):
    app = build_prototype(write_back_enabled=True)

    async def invoke(_request, _context):
        app.model_provider.calls += 1
        return ModelInvocationResult(provider_output, Usage(1, 3, 1))

    app.model_provider.invoke = invoke
    decision = app.admit_bulk_task(
        "bulk-job", ("item-1",), TaskFunction.CLASSIFY,
        freeze({"labels": ("ok",)}),
        source_rows=freeze({"item-1": {"text": "source"}}),
    )

    result = app.run_bulk(decision)
    terminal = app.repository.terminal_items("bulk-job")[0]

    assert result.classification == "failed"
    assert terminal.state is TerminalWorkItemState.VALIDATION_FAILED
    assert terminal.failure_code == expected_code
    assert terminal.result_reference is not None
    assert result.write_back_statuses == ()
    assert result.mutation_count == 0


def test_interactive_task_validation_failure_reports_exact_pack_and_withholds_output():
    app = build_prototype()

    async def invoke(request, _context):
        app.model_provider.calls += 1
        item_id = request.operation.work.input_ids[0]
        return ModelInvocationResult(({"id": item_id, "label": "not-allowed"},), Usage(1, 3, 1))

    app.model_provider.invoke = invoke
    response = app.admit_interactive_task(
        "invalid-label", "customers", 1.0, TaskFunction.CLASSIFY,
        freeze({"labels": ("allowed",)}),
    ).outcome

    failure = response.incompleteness[0]
    assert response.terminal_reason is InteractiveTerminalReason.VALIDATION_FAILED
    assert failure.operation_id == _task_operation(app, TaskFunction.CLASSIFY)
    assert failure.details["reason_codes"] == ("label_not_allowed",)
    assert tuple(item.operation_id for item in response.results) == ("query:customers",)
    assert app.budget.balance("prototype").actual == Usage(1, 3, 1)
    assert app.effect_adapter.mutation_count == 0


def test_interactive_security_denial_is_audited_before_provider_and_remains_read_only():
    app = build_prototype()
    app.model_executor._invoker._policies = app.model_executor._invoker._policies.__class__()

    decision = app.admit_interactive_task(
        "security-denied", "customers", 1.0, TaskFunction.SUMMARIZE,
        freeze({"max_words": 1}),
    )

    response = decision.outcome
    assert response.terminal_reason is InteractiveTerminalReason.SECURITY_REJECTED
    assert response.incompleteness[0].operation_id == _task_operation(app, TaskFunction.SUMMARIZE)
    assert response.incompleteness[0].details["reason_code"] == "security_policy_unavailable"
    assert app.model_provider.calls == 0
    balance = app.budget.balance("prototype")
    assert balance.reserved == Usage()
    assert balance.actual.cost_minor_units == 1
    assert balance.actual.total_tokens > 0
    assert app.effect_adapter.mutation_count == 0
    security = _events(app, "security_decision")
    assert len(security) == 1
    assert security[0]["correlation_id"] == str(decision.context.correlation_id)
    assert security[0]["details"]["reason_code"] == "policy_unavailable"

    bulk = build_prototype(write_back_enabled=True)
    bulk.model_executor._invoker._policies = bulk.model_executor._invoker._policies.__class__()
    bulk_decision = bulk.admit_bulk_task(
        "bulk-job", ("item-1",), TaskFunction.CLASSIFY,
        freeze({"labels": ("ok",)}),
    )
    bulk_result = bulk.run_bulk(bulk_decision)
    bulk_terminal = bulk.repository.terminal_items("bulk-job")[0]
    assert bulk_result.classification == "failed"
    assert bulk_terminal.state is TerminalWorkItemState.POLICY_REJECTED
    assert bulk_terminal.failure_code == "security_policy_unavailable"
    assert bulk.model_provider.calls == 0
    assert bulk_result.mutation_count == 0
    assert len(_events(bulk, "security_decision")) == 1


def test_budget_denial_preserves_interactive_fallback_and_bulk_budget_terminal_classification():
    interactive = build_prototype()
    interactive.budget = InMemoryBudgetController((BudgetLimit("prototype", 0, 0),))
    interactive.budget_adapter._controller = interactive.budget
    interactive.router._interactive_dispatcher._controller = interactive.budget

    response = interactive.admit_interactive_task(
        "budget-denied", "customers", 1.0, TaskFunction.CLASSIFY,
        freeze({"labels": ("ok",)}),
    ).outcome

    assert response.terminal_reason is InteractiveTerminalReason.BUDGET_EXHAUSTED
    assert response.incompleteness[0].details["exhausted_scopes"] == ("prototype",)
    assert interactive.model_provider.calls == 0
    assert interactive.effect_adapter.mutation_count == 0

    bulk = build_prototype(write_back_enabled=True)
    bulk.budget = InMemoryBudgetController((BudgetLimit("prototype", 0, 0),))
    bulk.budget_adapter._controller = bulk.budget
    bulk.router._bulk_dispatcher._controller = bulk.budget
    decision = bulk.admit_bulk_task(
        "bulk-job", ("item-1",), TaskFunction.CLASSIFY,
        freeze({"labels": ("ok",)}),
    )
    result = bulk.run_bulk(decision)
    terminal = bulk.repository.terminal_items("bulk-job")[0]

    assert result.classification == "budget-exhausted"
    assert terminal.state is TerminalWorkItemState.BUDGET_EXHAUSTED
    assert terminal.failure_code == "budget_exhausted"
    assert bulk.model_provider.calls == 0
    assert result.mutation_count == 0


def test_interactive_deadline_cancels_dependencies_suppresses_late_result_and_never_mutates():
    app = build_prototype()
    cancellation_tokens = []

    async def slow_invoke(request, _context):
        app.model_provider.calls += 1
        try:
            await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            await asyncio.sleep(0.05)
        item_id = request.operation.work.input_ids[0]
        return ModelInvocationResult(({"id": item_id, "label": "ok"},), Usage(1, 3, 1))

    async def cancel(token):
        cancellation_tokens.append(token)

    app.model_provider.invoke = slow_invoke
    app.model_executor._cancellation = cancel

    # Admission owns context construction; replace only its synchronous dispatch seam while retaining
    # the fully composed coordinator, then await it inside the running loop.
    dispatcher = app.router._interactive_dispatcher
    coordinator = dispatcher._coordinator
    captured = {}

    def capture(request, context):
        captured["request"], captured["context"] = request, context
        return object()

    dispatcher._coordinator = type("Capture", (), {"dispatch": staticmethod(capture)})()
    decision = app.admit_interactive_task(
        "deadline", "customers", 0.04, TaskFunction.CLASSIFY,
        freeze({"labels": ("ok",)}),
    )
    dispatcher._coordinator = coordinator

    async def execute_and_drain():
        response = await coordinator.execute(captured["request"], captured["context"])
        await asyncio.sleep(0.06)
        return response

    response = asyncio.run(execute_and_drain())

    assert decision.path.value == "interactive"
    assert response.terminal_reason is InteractiveTerminalReason.DEADLINE_EXCEEDED
    assert tuple(item.operation_id for item in response.results) == ("query:customers",)
    assert response.incompleteness[0].operation_id == _task_operation(app, TaskFunction.CLASSIFY)
    assert app.read_adapter.cancelled == [captured["context"].cancellation.token]
    assert cancellation_tokens == [captured["context"].cancellation.token]
    assert len(_events(app, "late_completion")) == 1
    assert app.effect_adapter.mutation_count == 0


def test_bulk_interruption_resumes_from_checkpoint_without_reinvocation_and_isolates_failure():
    app = build_prototype(write_back_enabled=False)
    app.model_provider.invalid_item_ids.add("item-2")
    decision = app.admit_bulk_task(
        "bulk-job", ("item-1", "item-2"), TaskFunction.SUMMARIZE,
        freeze({"max_words": 2}),
        source_rows=freeze({
            "item-1": {"text": "one two three"},
            "item-2": {"text": "four five six"},
        }),
    )

    first = app.worker.resume(
        "bulk-job", "bulk-job:item-1", "worker-before-crash",
        decision.context, max_stages=2,
    )
    calls_at_interrupt = app.model_provider.calls
    result = app.run_bulk(decision)

    assert tuple(stage.checkpoint_stage for stage in first) == (
        CheckpointStage.ACCEPTED, CheckpointStage.MODEL_COMPLETED,
    )
    assert calls_at_interrupt == 1
    assert app.model_provider.calls == 2
    assert result.classification == "partially-succeeded"
    assert dict(result.states) == {
        "item-1": "succeeded", "item-2": "validation-failed",
    }
    assert [checkpoint.completed_stage for checkpoint in app.repository.checkpoints(
        "bulk-job", "bulk-job:item-1"
    )] == list(CheckpointStage)[:-1]
    failed = next(item for item in app.repository.terminal_items("bulk-job") if item.item_id == "item-2")
    assert failed.state is TerminalWorkItemState.VALIDATION_FAILED
    assert dict(result.write_back_statuses) == {"item-1": "persisted"}
    assert result.mutation_count == 0
    assert result.telemetry_count > 0


@pytest.mark.parametrize(
    "function,parameters,field,expected",
    (
        (TaskFunction.CLASSIFY, freeze({"labels": ("ok",)}), "label", "ok"),
        (TaskFunction.SUMMARIZE, freeze({"max_words": 1}), "summary", "source"),
    ),
)
def test_bulk_disabled_write_back_checkpoints_canonical_output_without_mutation(
    function, parameters, field, expected,
):
    app = build_prototype(write_back_enabled=False)
    decision = app.admit_bulk_task(
        "bulk-job", ("item-1",), function, parameters,
        source_rows=freeze({"item-1": {"text": "source words"}}),
    )

    result = app.run_bulk(decision)
    terminal = app.repository.terminal_items("bulk-job")[0]
    accepted = json.loads(app.objects.get(terminal.result_reference))

    assert result.classification == "succeeded"
    assert dict(result.write_back_statuses) == {"item-1": "persisted"}
    assert accepted == [{"id": "item-1", field: expected}]
    assert app.repository.effect("bulk-job", "bulk-job:item-1") is None
    assert result.mutation_count == 0


def test_bulk_approved_write_back_is_approval_scoped_idempotent_and_audited():
    app = build_prototype(write_back_enabled=True)
    decision = app.admit_bulk_task(
        "bulk-job", ("item-1",), TaskFunction.CLASSIFY,
        freeze({"labels": ("ok",)}),
    )

    result = app.run_bulk(decision, simulate_interrupt=True)
    repeated = app.run_bulk(decision)
    effect = app.repository.effect("bulk-job", "bulk-job:item-1")

    assert result.classification == repeated.classification == "succeeded"
    assert dict(result.write_back_statuses) == {"item-1": "committed"}
    assert app.effect_adapter.mutation_count == 1
    assert effect is not None
    assert effect.idempotency_key == "bulk-job:item-1"
    assert app.model_provider.calls == 1
    assert app.repository.resume_state("bulk-job", "bulk-job:item-1").next_stage is None
    audits = tuple(event for event in app.telemetry.events if type(event).__name__ == "WriteBackAuditEvent")
    assert len(audits) == 1
    assert audits[0].approval_reference == "approval-1"
    assert audits[0].policy_version == "write-policy-1"
    assert audits[0].outcome.value == "commit"


def test_bulk_duplicate_write_effect_recovers_missing_checkpoint_without_second_mutation():
    app = build_prototype(write_back_enabled=True)
    decision = app.admit_bulk_task(
        "bulk-job", ("item-1",), TaskFunction.CLASSIFY,
        freeze({"labels": ("ok",)}),
    )
    key = "bulk-job:item-1"

    app.worker.resume("bulk-job", key, "worker-before-effect", decision.context, max_stages=4)
    claim = app.repository.claim("bulk-job", key, "worker-effect", app.clock(), app.worker._lease_duration)
    prepared = app.pipeline._prepared_from_accepted_checkpoint(claim)
    accepted = app.pipeline._load_json(claim.item.result_reference)
    validation = app.pipeline._validate_responses(prepared, accepted, decision.context, accepted_flat=True)
    first = app.pipeline._write_accepted(claim, decision.context, validation, TaskFunction.CLASSIFY)
    assert first.outcome == "committed"
    assert app.effect_adapter.mutation_count == 1

    # Simulate a lost completed checkpoint while retaining the committed effect record.
    app.repository._checkpoints[("bulk-job", key)].pop()
    item = app.repository._items[("bulk-job", key)]
    app.repository._items[("bulk-job", key)] = item.__class__(
        item.job_id, item.item_id, item.idempotency_key, item.payload_reference,
        item.configuration, item.state.__class__.CHECKPOINTED, item.attempt_count,
        None, None, item.result_reference, item.revision + 1,
    )

    resumed = app.worker.resume("bulk-job", key, "worker-recovery", decision.context)

    assert resumed[-1].outcome == "recovered"
    assert app.effect_adapter.mutation_count == 1
    assert app.model_provider.calls == 1
    assert app.repository.resume_state("bulk-job", key).next_stage is None
    assert app.repository.checkpoints("bulk-job", key)[-1].completed_stage is CheckpointStage.COMPLETED
    assert app.pipeline.statuses["item-1"] == "recovered"
