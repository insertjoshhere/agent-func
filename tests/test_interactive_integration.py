"""Focused legacy and task-aware model planning compatibility tests."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from hypothesis import given, settings, strategies as st

from ai_retrieval.domain.budget import Usage
from ai_retrieval.domain.configuration import ConfigurationReference, ExecutionConfiguration
from ai_retrieval.domain.execution import CancellationContext, DeadlineContext, ExecutionContext, ExecutionPath
from ai_retrieval.domain.identifiers import CorrelationId, ExecutionId
from ai_retrieval.domain.immutable import freeze
from ai_retrieval.domain.model_routing import ModelCandidate
from ai_retrieval.domain.work import ModelWork
from ai_retrieval.interactive import CandidateCatalog, RoutedModelPlanner


class RecordingRouter:
    def __init__(self) -> None:
        self.operations = []

    def admit(self, operation, candidates, policy, now):
        self.operations.append(operation)
        return SimpleNamespace(
            admitted=True, candidate=candidates[0], lease=object(), failure=None
        )

    def complete(self, lease) -> None:
        pass


def execution_context() -> ExecutionContext:
    accepted = datetime(2025, 1, 1, tzinfo=timezone.utc)
    configuration = ExecutionConfiguration(
        ConfigurationReference("default", "config-1"),
        freeze({"interactive": {
            "tenant_id": "tenant-1",
            "estimated_model_tokens": 41,
            "estimated_output_tokens": 7,
        }}),
        "security-1",
        "rules-1",
    )
    return ExecutionContext(
        ExecutionId("execution-1"), CorrelationId("correlation-1"),
        ExecutionPath.INTERACTIVE, configuration,
        DeadlineContext(accepted, accepted + timedelta(seconds=1)),
        CancellationContext("cancel-1", 0),
    )


def planner(router: RecordingRouter) -> RoutedModelPlanner:
    candidate = ModelCandidate(
        "model-1", "provider-1", frozenset({"extract", "ai_classify"}),
        frozenset(), 3, 1, 5, 1.0,
    )
    catalog = CandidateCatalog(lambda work, context: (candidate,), lambda context: None)
    return RoutedModelPlanner(
        router, catalog, clock=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc)
    )


def test_model_work_retains_four_field_legacy_positional_and_keyword_construction() -> None:
    positional = ModelWork("extract", "payload://legacy", ("row-1",), frozenset({"extract"}))
    keyword = ModelWork(
        task_type="extract",
        payload_reference="payload://legacy",
        input_ids=("row-1",),
        required_capabilities=frozenset({"extract"}),
    )

    assert positional == keyword
    assert positional.task_definition_version is None
    assert positional.estimated_input_tokens is None
    assert positional.estimated_output_tokens is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("task_definition_version", " "),
        ("estimated_input_tokens", -1),
        ("estimated_input_tokens", True),
        ("estimated_output_tokens", 1.5),
    ],
)
def test_model_work_rejects_invalid_optional_task_metadata(field, value) -> None:
    arguments = {
        "task_type": "ai_classify",
        "payload_reference": "payload://task",
        "input_ids": ("row-1",),
        "required_capabilities": frozenset({"ai_classify"}),
        field: value,
    }
    with pytest.raises(ValueError):
        ModelWork(**arguments)


def test_legacy_planning_retains_configuration_estimates() -> None:
    router = RecordingRouter()
    work = ModelWork("extract", "payload://legacy", ("row-1",), frozenset({"extract"}))

    plan = planner(router).plan("model:0:extract", work, execution_context(), 12)

    assert plan.operation.work is work
    assert plan.operation.operation_id == "model:0:extract"
    assert plan.operation.estimated_total_tokens == 41
    assert plan.estimate == Usage(3, 41, 7)


def test_task_aware_planning_uses_pack_input_and_output_estimates() -> None:
    router = RecordingRouter()
    work = ModelWork(
        "ai_classify", "payload://task", ("row-1",), frozenset({"ai_classify"}),
        "definition-1", 11, 5,
    )

    plan = planner(router).plan("model:0:ai_classify", work, execution_context(), 12)

    assert plan.operation.work is work
    assert plan.operation.operation_id == "model:0:ai_classify"
    assert plan.operation.estimated_total_tokens == 16
    assert plan.estimate == Usage(3, 11, 5)


def test_partially_supplied_task_estimates_fall_back_per_dimension() -> None:
    router = RecordingRouter()
    work = ModelWork(
        "ai_classify", "payload://task", ("row-1",), frozenset({"ai_classify"}),
        "definition-1", estimated_input_tokens=11,
    )

    plan = planner(router).plan("model:0:ai_classify", work, execution_context(), 12)

    assert plan.operation.estimated_total_tokens == 18
    assert plan.estimate == Usage(3, 11, 7)


_LEGACY_IDENTIFIER = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789-", min_size=1, max_size=12,
)


@settings(max_examples=100, deadline=None)
@given(
    payload_suffix=_LEGACY_IDENTIFIER,
    input_ids=st.lists(
        _LEGACY_IDENTIFIER, min_size=1, max_size=3, unique=True,
    ).map(tuple),
    model_valid=st.booleans(),
    cli_path=st.sampled_from(("interactive", "bulk")),
)
def test_property_42_legacy_model_work_compatibility(
    payload_suffix, input_ids, model_valid, cli_path,
) -> None:
    """**Validates: Requirements 18.8, 18.10**"""
    import io
    import json
    from contextlib import redirect_stdout

    from ai_retrieval.cli.app import main
    from ai_retrieval.composition import build_prototype
    from ai_retrieval.domain.admission import AdmissionEnvelope
    from ai_retrieval.domain.work import InteractiveRequest, QueryWork
    from ai_retrieval.interactive import InteractiveTerminalReason

    task_type = "extract"
    capability = frozenset({task_type})
    payload_reference = f"payload://{payload_suffix}"
    positional = ModelWork(task_type, payload_reference, input_ids, capability)
    keyword = ModelWork(
        task_type=task_type, payload_reference=payload_reference,
        input_ids=input_ids, required_capabilities=capability,
    )
    assert positional == keyword
    assert (
        positional.task_definition_version,
        positional.estimated_input_tokens,
        positional.estimated_output_tokens,
    ) == (None, None, None)

    router = RecordingRouter()
    context = execution_context()
    operation_id = f"model:0:{task_type}"
    plan = planner(router).plan(operation_id, positional, context, 12)
    assert plan.operation.operation_id == operation_id
    assert plan.operation.work is positional
    assert plan.operation.tenant_id == "tenant-1"
    assert plan.operation.scope_id == "execution-1"
    assert plan.operation.path is ExecutionPath.INTERACTIVE
    assert plan.operation.deadline == context.timing.deadline
    assert plan.operation.completion_reserve_ms == 12
    assert plan.operation.estimated_total_tokens == 41
    assert plan.estimate == Usage(3, 41, 7)

    app = build_prototype(model_valid=model_valid)
    request = InteractiveRequest(
        f"request-{payload_suffix}", QueryWork("customers"), (positional,),
    )
    response = app.router.admit(AdmissionEnvelope(
        app.reference, interactive=request, interactive_deadline_seconds=1.0,
    )).outcome
    assert app.model_provider.calls == 1
    assert app.effect_adapter.mutation_count == 0
    assert response.terminal_reason is (
        InteractiveTerminalReason.COMPLETE
        if model_valid else InteractiveTerminalReason.VALIDATION_FAILED
    )
    observed = response.results if model_valid else response.incompleteness
    assert tuple(item.operation_id for item in observed if item.operation_id != "query:customers") == (
        operation_id,
    )

    if cli_path == "interactive":
        command = [
            "interactive", "--request-id", f"cli-{payload_suffix}",
            "--query-plan", "customers", "--config-version", "legacy-v1",
        ]
        expected_keys = {"query_plan", "request_id"}
    else:
        command = [
            "bulk", "--job-id", f"cli-{payload_suffix}", f"--item={input_ids[0]}",
            "--config-version", "legacy-v1",
        ]
        expected_keys = {"item_count", "job_id"}
    output_buffer = io.StringIO()
    with redirect_stdout(output_buffer):
        exit_code = main(command)
    output = json.loads(output_buffer.getvalue())
    assert exit_code == 0
    assert output["path"] == cli_path
    assert set(output["outcome"]) == expected_keys
    assert not ({"task", "task_results", "task_failures"} & set(output["outcome"]))
