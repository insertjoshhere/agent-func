"""Focused tests for content-addressed immutable task definitions."""

from dataclasses import replace
from hashlib import sha256

import pytest
from hypothesis import given, settings, strategies as st

from ai_retrieval.domain.immutable import freeze
from ai_retrieval.tasks import (
    PackingLimits,
    SummaryLengthLimits,
    TaskDefinition,
    TaskDefinitionRegistry,
    TaskDefinitionValidationError,
    TaskFunction,
    build_seeded_task_definition_registry,
    canonical_definition_bytes,
)
from ai_retrieval.validation import FieldRule, FieldType, ValidationRules, ValidationRulesRegistry


def rules_registry() -> ValidationRulesRegistry:
    rules = ValidationRules("rules-1", "id", (FieldRule("schema.id", "id", FieldType.STRING),))
    return ValidationRulesRegistry((rules,))


def candidate(function: TaskFunction = TaskFunction.CLASSIFY, version: str = "") -> TaskDefinition:
    from ai_retrieval.tasks import task_output_schema

    return TaskDefinition(
        function, version, "Process canonical rows",
        freeze({"type": "object"}), freeze({"required": ["rows"]}),
        task_output_schema(function), frozenset({function.value}), "rules-1",
        PackingLimits(2, 100, 50, 4096), 4096,
        label_count_limit=4 if function is TaskFunction.CLASSIFY else None,
        label_character_limit=20 if function is TaskFunction.CLASSIFY else None,
        summary_limits=(SummaryLengthLimits(1000, 100, 2000) if function is TaskFunction.SUMMARIZE else None),
    )


def test_registration_assigns_canonical_sha256_and_resolves_exact_function_version() -> None:
    registry = TaskDefinitionRegistry(rules_registry())
    registered = registry.register(candidate())

    assert registered.version == sha256(canonical_definition_bytes(candidate())).hexdigest()
    assert registry.resolve(TaskFunction.CLASSIFY, registered.version) is registered
    assert registry.resolve(TaskFunction.SUMMARIZE, registered.version) is None
    assert registry.register(registered) is registered


def test_registration_aggregates_invalid_fields_and_unavailable_rules_without_mutation() -> None:
    registry = TaskDefinitionRegistry(rules_registry())
    invalid = replace(
        candidate(), prompt_template=" ", parameter_contract=freeze({}),
        required_capabilities=frozenset(), validation_rules_version="missing",
        packing_limits=PackingLimits(0, -1, 0, 0), response_size_limit=0,
        label_count_limit=None, label_character_limit=0,
    )

    with pytest.raises(TaskDefinitionValidationError) as raised:
        registry.register(invalid)

    assert set(raised.value.failed_rule_ids) >= {
        "task_definition.prompt_template", "task_definition.parameter_contract",
        "task_definition.required_capabilities", "task_definition.validation_rules_unavailable",
        "task_definition.packing_limits.max_items", "task_definition.packing_limits.max_input_tokens",
        "task_definition.packing_limits.max_output_tokens", "task_definition.packing_limits.max_payload_bytes",
        "task_definition.response_size_limit", "task_definition.label_count_limit",
        "task_definition.label_character_limit",
    }
    assert registry.resolve(TaskFunction.CLASSIFY, sha256(canonical_definition_bytes(invalid)).hexdigest()) is None


def test_supplied_version_must_match_content_and_registered_content_cannot_be_replaced() -> None:
    registry = TaskDefinitionRegistry(rules_registry())
    first = registry.register(candidate())

    with pytest.raises(TaskDefinitionValidationError) as mismatch:
        registry.register(replace(candidate(), version="0" * 64))
    assert "task_definition.version" in mismatch.value.failed_rule_ids

    changed = replace(candidate(), version=first.version, prompt_template="Changed instructions")
    with pytest.raises(TaskDefinitionValidationError) as collision:
        registry.register(changed)
    assert set(collision.value.failed_rule_ids) >= {
        "task_definition.version", "task_definition.version_immutable"
    }
    assert registry.resolve(TaskFunction.CLASSIFY, first.version) is first


def test_later_registration_uses_a_new_version_without_changing_prior_resolution() -> None:
    registry = TaskDefinitionRegistry(rules_registry())
    first = registry.register(candidate())
    second = registry.register(replace(candidate(), prompt_template="Later definition"))

    assert first.version != second.version
    assert registry.resolve(TaskFunction.CLASSIFY, first.version) is first
    assert registry.resolve(TaskFunction.CLASSIFY, second.version) is second


def test_seed_factory_registers_valid_classify_and_summarize_definitions() -> None:
    registry = build_seeded_task_definition_registry(rules_registry(), "rules-1")
    assert len(registry.registered_definitions) == 2
    seed_versions = []
    for function in TaskFunction:
        definition = next(
            item for item in registry.registered_definitions if item.function is function
        )
        assert registry.resolve(function, definition.version) is definition
        assert definition.validation_rules_version == "rules-1"
        assert function.value in definition.required_capabilities
        seed_versions.append(definition.version)
    assert len(set(seed_versions)) == 2


def test_seed_factory_fails_when_referenced_validation_rules_are_unavailable() -> None:
    with pytest.raises(TaskDefinitionValidationError) as raised:
        build_seeded_task_definition_registry(ValidationRulesRegistry(), "missing")
    assert raised.value.failed_rule_ids == ("task_definition.validation_rules_unavailable",)


class PreparationSpy:
    def __init__(self, result=None) -> None:
        self.calls = []
        self.result = result

    def prepare(self, resolved, context):
        self.calls.append((resolved, context))
        return self.result


def execution_context(bindings=...):
    from datetime import datetime, timezone

    from ai_retrieval.domain.configuration import ConfigurationReference, ExecutionConfiguration
    from ai_retrieval.domain.execution import CancellationContext, DeadlineContext, ExecutionContext, ExecutionPath
    from ai_retrieval.domain.identifiers import CorrelationId, ExecutionId

    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    content = {} if bindings is ... else {"task_definitions": bindings}
    configuration = ExecutionConfiguration(
        ConfigurationReference("default", "config-1"), freeze(content)
    )
    return ExecutionContext(
        ExecutionId("execution-1"), CorrelationId("correlation-1"),
        ExecutionPath.INTERACTIVE, configuration, DeadlineContext(now, None),
        CancellationContext("cancel-1", 0),
    )


def invocation(function=TaskFunction.CLASSIFY, requested_version=None):
    from ai_retrieval.tasks import RowInput, TaskInvocation

    parameters = {"labels": ["yes"]} if function is TaskFunction.CLASSIFY else {"max_words": 10}
    return TaskInvocation(
        function, freeze(parameters), (RowInput("row-1", freeze({"text": "value"})),),
        requested_definition_version=requested_version,
    )


def test_runtime_resolves_only_execution_bound_definition_and_preserves_it() -> None:
    from ai_retrieval.tasks import ResolvedTask, TaskRuntime

    registry = TaskDefinitionRegistry(rules_registry())
    bound = registry.register(candidate())
    context = execution_context({TaskFunction.CLASSIFY.value: bound.version})
    later = registry.register(replace(candidate(), prompt_template="Later definition"))
    spy = PreparationSpy()
    runtime = TaskRuntime(registry, spy)

    resolved = runtime.resolve(invocation(requested_version=bound.version), context)

    assert isinstance(resolved, ResolvedTask)
    assert resolved.definition is bound
    assert resolved.definition is not later
    assert resolved.invocation.requested_definition_version == bound.version
    assert spy.calls == []


def test_runtime_accepts_absent_caller_version_hint_and_delegates_resolved_definition() -> None:
    from ai_retrieval.tasks import ResolvedTask, TaskRuntime

    registry = TaskDefinitionRegistry(rules_registry())
    bound = registry.register(candidate())
    context = execution_context({TaskFunction.CLASSIFY.value: bound.version})
    sentinel = object()
    spy = PreparationSpy(sentinel)
    runtime = TaskRuntime(registry, spy)

    result = runtime.prepare(invocation(), context)

    assert result is sentinel
    assert len(spy.calls) == 1
    resolved, called_context = spy.calls[0]
    assert isinstance(resolved, ResolvedTask)
    assert resolved.definition is bound
    assert called_context is context


@pytest.mark.parametrize(
    ("bindings", "expected_reason"),
    [
        (..., "binding_missing"),
        (None, "binding_malformed"),
        ({TaskFunction.CLASSIFY.value: 7}, "binding_malformed"),
        ({TaskFunction.CLASSIFY.value: " "}, "binding_malformed"),
    ],
)
def test_runtime_rejects_missing_or_malformed_binding_before_preparation_effects(
    bindings, expected_reason
) -> None:
    from ai_retrieval.tasks import TaskFailure, TaskFailureCode, TaskFailureStage, TaskRuntime

    registry = TaskDefinitionRegistry(rules_registry())
    spy = PreparationSpy()
    runtime = TaskRuntime(registry, spy)

    failure = runtime.prepare(invocation(), execution_context(bindings))

    assert isinstance(failure, TaskFailure)
    assert failure.stage is TaskFailureStage.RESOLUTION
    assert failure.code is TaskFailureCode.TASK_DEFINITION_UNAVAILABLE
    assert failure.failed_rule_ids == ("task_definition.binding",)
    assert failure.details["reason"] == expected_reason
    assert spy.calls == []


def test_runtime_rejects_unavailable_bound_definition_before_preparation_effects() -> None:
    from ai_retrieval.tasks import TaskFailure, TaskFailureCode, TaskRuntime

    registry = TaskDefinitionRegistry(rules_registry())
    spy = PreparationSpy()
    missing_version = "0" * 64

    failure = TaskRuntime(registry, spy).prepare(
        invocation(), execution_context({TaskFunction.CLASSIFY.value: missing_version})
    )

    assert isinstance(failure, TaskFailure)
    assert failure.code is TaskFailureCode.TASK_DEFINITION_UNAVAILABLE
    assert failure.failed_rule_ids == ("task_definition.available",)
    assert failure.details["bound_version"] == missing_version
    assert failure.details["reason"] == "definition_unavailable"
    assert spy.calls == []


def test_runtime_rejects_incompatible_caller_version_before_preparation_effects() -> None:
    from ai_retrieval.tasks import TaskFailure, TaskFailureCode, TaskRuntime

    registry = TaskDefinitionRegistry(rules_registry())
    bound = registry.register(candidate())
    requested = "f" * 64
    spy = PreparationSpy()

    failure = TaskRuntime(registry, spy).prepare(
        invocation(requested_version=requested),
        execution_context({TaskFunction.CLASSIFY.value: bound.version}),
    )

    assert isinstance(failure, TaskFailure)
    assert failure.code is TaskFailureCode.TASK_DEFINITION_INCOMPATIBLE
    assert failure.failed_rule_ids == ("task_definition.requested_version",)
    assert failure.details["bound_version"] == bound.version
    assert failure.details["requested_version"] == requested
    assert failure.details["reason"] == "requested_version_mismatch"
    assert spy.calls == []


def test_runtime_rejects_registry_result_incompatible_with_bound_pair() -> None:
    from ai_retrieval.tasks import TaskFailure, TaskFailureCode, TaskRuntime

    bound = TaskDefinitionRegistry(rules_registry()).register(candidate())
    incompatible = replace(bound, version="e" * 64)

    class IncompatibleLookup:
        def resolve(self, function, version):
            return incompatible

    spy = PreparationSpy()
    failure = TaskRuntime(IncompatibleLookup(), spy).prepare(
        invocation(), execution_context({TaskFunction.CLASSIFY.value: bound.version})
    )

    assert isinstance(failure, TaskFailure)
    assert failure.code is TaskFailureCode.TASK_DEFINITION_INCOMPATIBLE
    assert failure.failed_rule_ids == ("task_definition.compatible",)
    assert failure.details["reason"] == "resolved_definition_mismatch"
    assert spy.calls == []


_NONBLANK_TEXT = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)), min_size=1, max_size=24
).filter(str.strip)


@st.composite
def task_definitions(draw):
    function = draw(st.sampled_from(tuple(TaskFunction)))
    prompt = draw(_NONBLANK_TEXT)
    contract_value = draw(st.integers(min_value=-100, max_value=100))
    limits = PackingLimits(
        draw(st.integers(min_value=1, max_value=100)),
        draw(st.integers(min_value=1, max_value=20_000)),
        draw(st.integers(min_value=1, max_value=20_000)),
        draw(st.integers(min_value=1, max_value=2_000_000)),
    )
    common = {
        "function": function,
        "version": "",
        "prompt_template": prompt,
        "parameter_contract": freeze({"value": contract_value, "type": "object"}),
        "payload_contract": freeze({"required": ["rows"], "type": "object"}),
        "output_schema": freeze({"items": {"type": "object"}, "type": "array"}),
        "required_capabilities": frozenset({function.value}),
        "validation_rules_version": "rules-1",
        "packing_limits": limits,
        "response_size_limit": draw(st.integers(min_value=1, max_value=2_000_000)),
    }
    if function is TaskFunction.CLASSIFY:
        return TaskDefinition(
            **common,
            label_count_limit=draw(st.integers(min_value=1, max_value=100)),
            label_character_limit=draw(st.integers(min_value=1, max_value=1_000)),
        )
    return TaskDefinition(
        **common,
        summary_limits=SummaryLengthLimits(
            draw(st.integers(min_value=1, max_value=100_000)),
            draw(st.integers(min_value=1, max_value=2_000)),
            draw(st.integers(min_value=1, max_value=20_000)),
        ),
    )


@settings(max_examples=100, deadline=None)
@given(definition=task_definitions())
def test_property_31_complete_immutable_task_definition_registration(
    definition: TaskDefinition,
) -> None:
    """**Validates: Requirements 14.1, 14.2, 14.3, 14.4**"""
    registry = TaskDefinitionRegistry(rules_registry())
    expected_version = sha256(canonical_definition_bytes(definition)).hexdigest()

    registered = registry.register(definition)
    equivalent = replace(definition, version=expected_version)

    assert registered.version == expected_version
    assert canonical_definition_bytes(equivalent) == canonical_definition_bytes(definition)
    assert registry.register(equivalent) is registered
    assert registry.resolve(definition.function, expected_version) is registered

    changed = replace(
        definition,
        version=expected_version,
        prompt_template=definition.prompt_template + " changed",
    )
    before_collision = registry.registered_definitions
    with pytest.raises(TaskDefinitionValidationError) as raised:
        registry.register(changed)

    assert set(raised.value.failed_rule_ids) >= {
        "task_definition.version",
        "task_definition.version_immutable",
    }
    assert registry.registered_definitions == before_collision
    assert registry.resolve(definition.function, expected_version) is registered


@settings(max_examples=100, deadline=None)
@given(
    function=st.sampled_from(tuple(TaskFunction)),
    later_prompt=_NONBLANK_TEXT,
    resolution_case=st.sampled_from(
        ("bound", "bound_with_hint", "missing", "unavailable", "incompatible")
    ),
)
def test_property_32_execution_bound_task_definition_resolution(
    function: TaskFunction,
    later_prompt: str,
    resolution_case: str,
) -> None:
    """**Validates: Requirements 14.5, 14.6, 14.7**"""
    from ai_retrieval.tasks import ResolvedTask, TaskFailure, TaskFailureCode, TaskRuntime

    registry = TaskDefinitionRegistry(rules_registry())
    bound = registry.register(candidate(function))
    later = registry.register(
        replace(candidate(function), prompt_template="later: " + later_prompt)
    )
    spy = PreparationSpy(object())
    runtime = TaskRuntime(registry, spy)

    if resolution_case == "missing":
        context = execution_context({})
        requested = None
    elif resolution_case == "unavailable":
        context = execution_context({function.value: "0" * 64})
        requested = None
    else:
        context = execution_context({function.value: bound.version})
        requested = (
            "f" * 64
            if resolution_case == "incompatible"
            else bound.version if resolution_case == "bound_with_hint" else None
        )

    resolved = runtime.resolve(invocation(function, requested), context)

    if resolution_case in {"bound", "bound_with_hint"}:
        assert isinstance(resolved, ResolvedTask)
        assert resolved.definition is bound
        assert resolved.definition is not later
        assert resolved.definition.version == bound.version
    else:
        assert isinstance(resolved, TaskFailure)
        expected_code = (
            TaskFailureCode.TASK_DEFINITION_INCOMPATIBLE
            if resolution_case == "incompatible"
            else TaskFailureCode.TASK_DEFINITION_UNAVAILABLE
        )
        assert resolved.code is expected_code
    assert spy.calls == []

    prepared = runtime.prepare(invocation(function, requested), context)
    if resolution_case in {"bound", "bound_with_hint"}:
        assert prepared is spy.result
        assert len(spy.calls) == 1
        prepared_resolution, prepared_context = spy.calls[0]
        assert prepared_resolution.definition is bound
        assert prepared_context is context
    else:
        assert isinstance(prepared, TaskFailure)
        assert spy.calls == []
