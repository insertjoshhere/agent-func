"""Focused invariants for immutable provider-neutral task models."""

from dataclasses import FrozenInstanceError

import pytest

from ai_retrieval.domain.immutable import FrozenMapping, freeze
from ai_retrieval.domain.work import ModelWork
from ai_retrieval.tasks import (
    PackedRequest,
    PackingLimits,
    PreparedTask,
    RowInput,
    SummaryLengthLimits,
    TaskDefinition,
    TaskFailure,
    TaskFailureCode,
    TaskFailureStage,
    TaskFunction,
    TaskInvocation,
)


def definition(function: TaskFunction = TaskFunction.CLASSIFY) -> TaskDefinition:
    return TaskDefinition(
        function=function,
        version="definition-1",
        prompt_template="Process canonical rows",
        parameter_contract=freeze({"type": "object"}),
        payload_contract=freeze({"required": ["rows"]}),
        output_schema=freeze({"type": "array"}),
        required_capabilities=frozenset({function.value}),
        validation_rules_version="rules-1",
        packing_limits=PackingLimits(2, 100, 50, 4096),
        response_size_limit=4096,
        label_count_limit=4 if function is TaskFunction.CLASSIFY else None,
        label_character_limit=20 if function is TaskFunction.CLASSIFY else None,
        summary_limits=(
            SummaryLengthLimits(1000, 100, 2000)
            if function is TaskFunction.SUMMARIZE else None
        ),
    )


def test_public_models_are_immutable_and_use_frozen_mappings() -> None:
    source = {"nested": {"values": [1, 2]}}
    row = RowInput("row-1", freeze(source))
    invocation = TaskInvocation(TaskFunction.CLASSIFY, freeze({"labels": ["yes", "no"]}), (row,))
    source["nested"]["values"].append(3)

    assert row.source_fields["nested"]["values"] == (1, 2)
    assert invocation.parameters["labels"] == ("yes", "no")
    with pytest.raises(FrozenInstanceError):
        row.identifier = "changed"
    with pytest.raises(TypeError, match="FrozenMapping"):
        RowInput("row-2", {"value": 1})  # type: ignore[arg-type]


def test_supported_function_values_are_exactly_classify_and_summarize() -> None:
    assert tuple(TaskFunction) == (TaskFunction.CLASSIFY, TaskFunction.SUMMARIZE)
    assert tuple(value.value for value in TaskFunction) == ("ai_classify", "ai_summarize")
    with pytest.raises(ValueError, match="ai_classify or ai_summarize"):
        TaskInvocation("ai_extract", freeze({}), (RowInput("row-1", freeze({})),))  # type: ignore[arg-type]


def test_invocation_requires_nonempty_unique_nonblank_ordered_rows() -> None:
    row = RowInput("row-1", freeze({"text": "value"}))
    with pytest.raises(ValueError, match="at least one row"):
        TaskInvocation(TaskFunction.CLASSIFY, freeze({}), ())
    with pytest.raises(ValueError, match="unique"):
        TaskInvocation(TaskFunction.CLASSIFY, freeze({}), (row, row))
    with pytest.raises(ValueError, match="must not be blank"):
        RowInput(" ", freeze({}))
    with pytest.raises(ValueError, match="version must not be blank"):
        TaskInvocation(TaskFunction.CLASSIFY, freeze({}), (row,), " ")

def test_prepared_task_requires_exact_ordered_partition_and_matching_model_work() -> None:
    rows = (RowInput("row-1", freeze({})), RowInput("row-2", freeze({})))
    invocation = TaskInvocation(TaskFunction.CLASSIFY, freeze({"labels": ["yes"]}), rows)
    packs = (
        PackedRequest(TaskFunction.CLASSIFY, "definition-1", 0, ("row-1",), b"one", 1, 1),
        PackedRequest(TaskFunction.CLASSIFY, "definition-1", 1, ("row-2",), b"two", 1, 1),
    )
    work = (
        ModelWork("ai_classify", "payload-1", ("row-1",), frozenset({"ai_classify"})),
        ModelWork("ai_classify", "payload-2", ("row-2",), frozenset({"ai_classify"})),
    )

    prepared = PreparedTask(definition(), invocation, packs, work)
    assert tuple(identifier for pack in prepared.packs for identifier in pack.input_ids) == (
        "row-1", "row-2"
    )
    with pytest.raises(ValueError, match="preserve invocation row order"):
        PreparedTask(definition(), invocation, tuple(reversed(packs)), work)
    with pytest.raises(ValueError, match="one ModelWork"):
        PreparedTask(definition(), invocation, packs, work[:1])


def test_packed_request_enforces_local_execution_invariants() -> None:
    with pytest.raises(ValueError, match="nonnegative integer"):
        PackedRequest(TaskFunction.CLASSIFY, "v1", True, ("row-1",), b"payload", 1, 1)
    with pytest.raises(ValueError, match="non-blank input"):
        PackedRequest(TaskFunction.CLASSIFY, "v1", 0, (), b"payload", 1, 1)
    with pytest.raises(ValueError, match="payload must not be empty"):
        PackedRequest(TaskFunction.CLASSIFY, "v1", 0, ("row-1",), b"", 1, 1)
    with pytest.raises(ValueError, match="nonnegative integers"):
        PackedRequest(TaskFunction.CLASSIFY, "v1", 0, ("row-1",), b"payload", -1, 1)


def test_task_failure_has_stable_values_and_redacted_immutable_details() -> None:
    failure = TaskFailure(
        TaskFailureStage.PARSING,
        TaskFailureCode.MALFORMED_STRUCTURED_OUTPUT,
        ("structured_output",),
        freeze({"representation": "json", "response_size": 12}),
    )

    assert failure.stage == "parsing"
    assert failure.code == "malformed_structured_output"
    assert isinstance(failure.details, FrozenMapping)
    with pytest.raises(ValueError, match="must be unique"):
        TaskFailure(
            TaskFailureStage.VALIDATION,
            TaskFailureCode.OUTPUT_SCHEMA_VIOLATION,
            ("schema", "schema"),
        )
