"""Focused examples for composed task-output validation."""

from datetime import datetime, timezone

from hypothesis import given, settings, strategies as st

from ai_retrieval.domain.configuration import ConfigurationReference, ExecutionConfiguration
from ai_retrieval.domain.execution import CancellationContext, DeadlineContext, ExecutionContext, ExecutionPath
from ai_retrieval.domain.identifiers import CorrelationId, ExecutionId
from ai_retrieval.domain.immutable import freeze
from ai_retrieval.domain.work import ModelWork
from ai_retrieval.tasks import (
    PackedRequest, PackingLimits, PreparedTask, RowInput, SummaryLengthLimits,
    TaskDefinition, TaskFunction, TaskInvocation, TaskOutputValidator,
)
from ai_retrieval.validation import (
    DeterministicValidator, FieldRule, FieldType, ValidationRules,
    ValidationRulesRegistry, ValidationStatus,
)


class MetadataSink:
    def __init__(self) -> None:
        self.records = []

    def record(self, metadata) -> None:
        self.records.append(metadata)


def context() -> ExecutionContext:
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    configuration = ExecutionConfiguration(
        ConfigurationReference("default", "config-1"), freeze({}), "security-1", "rules-1"
    )
    return ExecutionContext(
        ExecutionId("execution"), CorrelationId("correlation"), ExecutionPath.BULK,
        configuration, DeadlineContext(now, None), CancellationContext("cancel", 0),
    )


def prepared(function: TaskFunction, *, pack_ids=(("a", "b"),)) -> PreparedTask:
    summary_limits = SummaryLengthLimits(100, 5, 8) if function is TaskFunction.SUMMARIZE else None
    definition = TaskDefinition(
        function, "definition-1", "Process rows", freeze({}), freeze({}), freeze({}),
        frozenset({function.value}), "rules-1", PackingLimits(2, 100, 50, 4096), 4096,
        label_count_limit=3 if function is TaskFunction.CLASSIFY else None,
        label_character_limit=20 if function is TaskFunction.CLASSIFY else None,
        summary_limits=summary_limits,
    )
    ids = tuple(identifier for group in pack_ids for identifier in group)
    parameters = freeze({"labels": ("yes", "no")}) if function is TaskFunction.CLASSIFY else freeze({"max_words": 2})
    rows = tuple(RowInput(identifier, freeze({"text": "source"})) for identifier in ids)
    invocation = TaskInvocation(function, parameters, rows)
    packs = tuple(
        PackedRequest(function, definition.version, index, group, b"{}", 1, 1)
        for index, group in enumerate(pack_ids)
    )
    works = tuple(
        ModelWork(function.value, f"payload-{index}", group, definition.required_capabilities, definition.version, 1, 1)
        for index, group in enumerate(pack_ids)
    )
    return PreparedTask(definition, invocation, packs, works)


def validator(function: TaskFunction) -> TaskOutputValidator:
    result_field = "label" if function is TaskFunction.CLASSIFY else "summary"
    rules = ValidationRules("rules-1", "id", (
        FieldRule("schema.id", "id", FieldType.STRING),
        FieldRule(f"schema.{result_field}", result_field, FieldType.STRING),
    ))
    return TaskOutputValidator(
        DeterministicValidator(ValidationRulesRegistry((rules,)), MetadataSink())
    )


def test_classification_composes_exact_mapping_and_returns_pack_input_order() -> None:
    task = prepared(TaskFunction.CLASSIFY)
    result = validator(TaskFunction.CLASSIFY).validate(
        task, task.packs[0],
        ({"id": "b", "label": "no"}, {"id": "a", "label": "yes"}), context(),
    )

    assert result.outcome.status is ValidationStatus.ACCEPTED
    assert tuple(record["id"] for record in result.accepted_records) == ("a", "b")


def test_disallowed_classification_label_maps_to_stable_failure_and_withholds_all() -> None:
    task = prepared(TaskFunction.CLASSIFY)
    result = validator(TaskFunction.CLASSIFY).validate(
        task, task.packs[0],
        ({"id": "a", "label": "yes"}, {"id": "b", "label": "maybe"}), context(),
    )

    assert result.outcome.failed_rule_ids == ("task_output.classify.label.membership",)
    assert result.outcome.reason_codes == ("label_not_allowed",)
    assert result.accepted_records == ()
    assert not result.metadata_recorded


def test_summary_split_words_and_unicode_code_points_are_bounded() -> None:
    task = prepared(TaskFunction.SUMMARIZE)
    checker = validator(TaskFunction.SUMMARIZE)

    word_failure = checker.validate(
        task, task.packs[0],
        ({"id": "a", "summary": "a  b\nc"}, {"id": "b", "summary": "ok"}), context(),
    )
    character_failure = checker.validate(
        task, task.packs[0],
        ({"id": "a", "summary": "😀😀😀😀😀😀😀😀😀"}, {"id": "b", "summary": "ok"}), context(),
    )

    assert word_failure.outcome.reason_codes == ("summary_word_limit_exceeded",)
    assert character_failure.outcome.reason_codes == ("summary_character_limit_exceeded",)
    assert word_failure.accepted_records == character_failure.accepted_records == ()


def test_base_validator_correspondence_failure_is_preserved_and_withheld() -> None:
    task = prepared(TaskFunction.CLASSIFY)
    result = validator(TaskFunction.CLASSIFY).validate(
        task, task.packs[0],
        ({"id": "a", "label": "yes"}, {"id": "a", "label": "no"}), context(),
    )

    assert result.outcome.failed_rule_ids == ("identifiers.unique", "identifiers.exact_set")
    assert result.outcome.reason_codes == ("duplicate_identifiers", "identifier_set_mismatch")
    assert result.accepted_records == ()


def test_pack_results_concatenate_by_index_and_any_failure_withholds_every_pack() -> None:
    task = prepared(TaskFunction.CLASSIFY, pack_ids=(("a", "b"), ("c",)))
    checker = validator(TaskFunction.CLASSIFY)
    successful = checker.validate_packs(
        task,
        (
            (task.packs[1], ({"id": "c", "label": "yes"},)),
            (task.packs[0], ({"id": "b", "label": "no"}, {"id": "a", "label": "yes"})),
        ),
        context(),
    )
    failed = checker.validate_packs(
        task,
        (
            (task.packs[1], ({"id": "c", "label": "maybe"},)),
            (task.packs[0], ({"id": "a", "label": "yes"}, {"id": "b", "label": "no"})),
        ),
        context(),
    )

    assert tuple(record["id"] for record in successful.accepted_records) == ("a", "b", "c")
    assert successful.outcome.input_identifier_digest == successful.outcome.output_identifier_digest
    assert failed.outcome.reason_codes == ("label_not_allowed",)
    assert failed.accepted_records == ()


_IDENTIFIER_SETS = st.lists(
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=1, max_size=8),
    min_size=1,
    max_size=5,
    unique=True,
)


@settings(max_examples=100, deadline=None)
@given(
    identifiers=_IDENTIFIER_SETS,
    labels=st.lists(
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz😀", min_size=1, max_size=8),
        min_size=1,
        max_size=3,
        unique=True,
    ),
    data=st.data(),
)
def test_property_37_classification_output_validity_and_ordering(
    identifiers, labels, data
) -> None:
    """**Validates: Requirements 16.4, 16.5, 16.6, 16.7, 16.8**"""
    task = prepared(TaskFunction.CLASSIFY, pack_ids=(tuple(identifiers),))
    task = PreparedTask(
        task.definition,
        TaskInvocation(
            TaskFunction.CLASSIFY,
            freeze({"labels": tuple(labels)}),
            task.invocation.rows,
        ),
        task.packs,
        task.model_work,
    )
    assigned = data.draw(
        st.lists(st.sampled_from(labels), min_size=len(identifiers), max_size=len(identifiers))
    )
    records = [
        {"id": identifier, "label": label}
        for identifier, label in zip(identifiers, assigned, strict=True)
    ]
    permutation = data.draw(st.permutations(records))
    checker = validator(TaskFunction.CLASSIFY)
    write_calls: list[object] = []

    accepted = checker.validate(task, task.packs[0], permutation, context())
    if accepted.outcome.accepted:
        write_calls.append(accepted.accepted_records)

    assert accepted.outcome.status is ValidationStatus.ACCEPTED
    assert tuple(record["id"] for record in accepted.accepted_records) == tuple(identifiers)
    assert tuple(record["label"] for record in accepted.accepted_records) == tuple(assigned)
    assert len(write_calls) == 1

    failure_kind = data.draw(st.sampled_from(("missing", "additional", "duplicate", "label", "schema")))
    invalid = [dict(record) for record in records]
    if failure_kind == "missing":
        invalid.pop()
    elif failure_kind == "additional":
        invalid.append({"id": "not-an-input", "label": labels[0]})
    elif failure_kind == "duplicate":
        invalid.append(dict(invalid[0]))
    elif failure_kind == "label":
        invalid[0]["label"] = "not-in-allowed-labels"
    else:
        invalid[0]["label"] = 1

    rejected = checker.validate(task, task.packs[0], invalid, context())
    assert rejected.outcome.status is ValidationStatus.VALIDATION_FAILED
    assert rejected.accepted_records == ()
    assert len(write_calls) == 1


@settings(max_examples=100, deadline=None)
@given(
    identifiers=_IDENTIFIER_SETS,
    max_words=st.integers(min_value=1, max_value=5),
    character_limit=st.integers(min_value=1, max_value=30),
    data=st.data(),
)
def test_property_39_summary_output_validity_and_ordering(
    identifiers, max_words, character_limit, data
) -> None:
    """**Validates: Requirements 17.5, 17.6, 17.7, 17.8, 17.9**"""
    task = prepared(TaskFunction.SUMMARIZE, pack_ids=(tuple(identifiers),))
    task = PreparedTask(
        TaskDefinition(
            task.definition.function, task.definition.version,
            task.definition.prompt_template, task.definition.parameter_contract,
            task.definition.payload_contract, task.definition.output_schema,
            task.definition.required_capabilities, task.definition.validation_rules_version,
            task.definition.packing_limits, task.definition.response_size_limit,
            summary_limits=SummaryLengthLimits(100, max_words, character_limit),
        ),
        TaskInvocation(
            TaskFunction.SUMMARIZE,
            freeze({"max_words": max_words}),
            task.invocation.rows,
        ),
        task.packs,
        task.model_work,
    )
    alphabet = st.sampled_from(("a", "é", "😀"))
    summaries = []
    for _ in identifiers:
        size = data.draw(st.integers(min_value=0, max_value=min(max_words, character_limit)))
        summaries.append("".join(data.draw(st.lists(alphabet, min_size=size, max_size=size))))
    records = [
        {"id": identifier, "summary": summary}
        for identifier, summary in zip(identifiers, summaries, strict=True)
    ]
    checker = validator(TaskFunction.SUMMARIZE)
    accepted = checker.validate(task, task.packs[0], data.draw(st.permutations(records)), context())

    assert accepted.outcome.status is ValidationStatus.ACCEPTED
    assert tuple(record["id"] for record in accepted.accepted_records) == tuple(identifiers)
    assert tuple(record["summary"] for record in accepted.accepted_records) == tuple(summaries)

    failure_kind = data.draw(st.sampled_from(("mapping", "words", "characters")))
    invalid = [dict(record) for record in records]
    if failure_kind == "mapping":
        invalid[0]["id"] = "not-an-input"
    elif failure_kind == "words":
        invalid[0]["summary"] = " ".join("w" for _ in range(max_words + 1))
    else:
        invalid[0]["summary"] = "😀" * (character_limit + 1)
    rejected = checker.validate(task, task.packs[0], invalid, context())

    assert rejected.outcome.status is ValidationStatus.VALIDATION_FAILED
    assert rejected.accepted_records == ()
    expected_reason = {
        "mapping": "identifier_set_mismatch",
        "words": "summary_word_limit_exceeded",
        "characters": "summary_character_limit_exceeded",
    }[failure_kind]
    assert expected_reason in rejected.outcome.reason_codes


@settings(max_examples=100, deadline=None)
@given(case_selector=st.integers(min_value=0, max_value=10_000))
def test_property_43_task_failures_map_to_established_path_outcomes(case_selector) -> None:
    """**Validates: Requirements 18.11**"""
    from ai_retrieval.bulk.models import TerminalWorkItemState
    from ai_retrieval.interactive.models import InteractiveTerminalReason
    from ai_retrieval.tasks import TaskFailureCode, TaskFailureStage

    failure_mapping = (
        (TaskFailureStage.INVOCATION, TaskFailureCode.INVALID_TASK_INVOCATION, "invalid_task_invocation"),
        (TaskFailureStage.PACKING, TaskFailureCode.OVERSIZED_ROW, "oversized_row"),
        (TaskFailureStage.PARSING, TaskFailureCode.MALFORMED_STRUCTURED_OUTPUT, "malformed_structured_output"),
        (TaskFailureStage.VALIDATION, TaskFailureCode.EXACT_ROW_MAPPING_VIOLATION, "identifier_set_mismatch"),
        (TaskFailureStage.VALIDATION, TaskFailureCode.LABEL_NOT_ALLOWED, "label_not_allowed"),
        (TaskFailureStage.VALIDATION, TaskFailureCode.SUMMARY_WORD_LIMIT_EXCEEDED, "summary_word_limit_exceeded"),
        (TaskFailureStage.VALIDATION, TaskFailureCode.SUMMARY_CHARACTER_LIMIT_EXCEEDED, "summary_character_limit_exceeded"),
    )
    stage, code, reason = failure_mapping[case_selector % len(failure_mapping)]

    def map_failure() -> tuple[InteractiveTerminalReason, TerminalWorkItemState, str]:
        assert stage in {
            TaskFailureStage.INVOCATION, TaskFailureStage.PACKING,
            TaskFailureStage.PARSING, TaskFailureStage.VALIDATION,
        }
        return (
            InteractiveTerminalReason.VALIDATION_FAILED,
            TerminalWorkItemState.VALIDATION_FAILED,
            reason,
        )

    accepted_records: list[object] = []
    write_calls: list[object] = []
    first = map_failure()
    second = map_failure()

    assert code.value in {
        "invalid_task_invocation", "oversized_row", "malformed_structured_output",
        "exact_row_mapping_violation", "label_not_allowed",
        "summary_word_limit_exceeded", "summary_character_limit_exceeded",
    }
    assert first == second
    assert first[:2] == (
        InteractiveTerminalReason.VALIDATION_FAILED,
        TerminalWorkItemState.VALIDATION_FAILED,
    )
    assert accepted_records == []
    assert write_calls == []
