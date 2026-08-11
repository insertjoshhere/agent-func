import pytest

from ai_retrieval.composition import build_prototype
from ai_retrieval.domain.immutable import FrozenMapping, freeze
from ai_retrieval.interactive import InteractiveTerminalReason
from ai_retrieval.tasks import TaskFunction


def test_composed_interactive_completion_and_validation_fallback_are_read_only():
    complete_app = build_prototype()
    complete = complete_app.admit_interactive("request-complete", "customers", 1.0)

    assert complete.outcome.complete
    assert complete.outcome.terminal_reason is InteractiveTerminalReason.COMPLETE
    assert tuple(item.operation_id for item in complete.outcome.results) == (
        "query:customers", "model:0:extract",
    )
    assert complete_app.read_adapter.read_count == 1
    assert complete_app.model_provider.calls == 1
    assert complete_app.effect_adapter.mutation_count == 0
    assert complete_app.telemetry.events

    fallback_app = build_prototype(model_valid=False)
    fallback = fallback_app.admit_interactive("request-fallback", "customers", 1.0)

    assert not fallback.outcome.complete
    assert fallback.outcome.terminal_reason is InteractiveTerminalReason.VALIDATION_FAILED
    assert tuple(item.operation_id for item in fallback.outcome.incompleteness) == ("model:0:extract",)
    assert fallback_app.effect_adapter.mutation_count == 0


def test_composed_bulk_resumes_and_write_disabled_persists_without_mutation():
    app = build_prototype(write_back_enabled=False)
    decision = app.admit_bulk("bulk-job", ("item-1",))

    result = app.run_bulk(decision, simulate_interrupt=True)

    assert result.classification == "succeeded"
    assert dict(result.states) == {"item-1": "succeeded"}
    assert dict(result.attempts)["item-1"] > 1
    assert dict(result.write_back_statuses) == {"item-1": "persisted"}
    assert result.mutation_count == 0
    assert result.telemetry_count > 0


def test_composed_bulk_approved_write_commits_one_effect():
    app = build_prototype(write_back_enabled=True)
    decision = app.admit_bulk("bulk-job", ("item-1",))

    result = app.run_bulk(decision)

    assert result.classification == "succeeded"
    assert dict(result.write_back_statuses) == {"item-1": "committed"}
    assert result.mutation_count == 1
    assert app.repository.effect("bulk-job", "bulk-job:item-1") is not None


def test_composed_task_dependencies_and_exact_versions_are_bound_immutably():
    app = build_prototype()
    configuration = app.configuration.bind(app.reference)
    bindings = configuration.content["task_definitions"]

    assert isinstance(bindings, FrozenMapping)
    assert bindings == app.task_definition_versions
    assert set(bindings) == {function.value for function in TaskFunction}
    for definition in app.task_registry.registered_definitions:
        assert bindings[definition.function.value] == definition.version
        assert app.task_registry.resolve(definition.function, definition.version) is definition
        rules = app.validation_registry.resolve(definition.validation_rules_version)
        assert rules is not None
        assert {rule.field for rule in rules.field_rules} == {"id", "label", "summary"}

    assert app.token_estimator is not None
    assert app.payload_store is not None
    assert app.task_runtime is not None
    assert app.task_parser is not None
    assert app.task_validator is not None


@pytest.mark.parametrize(
    "function,parameters,field,expected",
    (
        (TaskFunction.CLASSIFY, freeze({"labels": ("approved", "rejected")}), "label", "approved"),
        (TaskFunction.SUMMARIZE, freeze({"max_words": 1}), "summary", "Ada"),
    ),
)
def test_composed_interactive_task_functions_use_function_aware_provider_and_validation(
    function, parameters, field, expected,
):
    app = build_prototype()

    decision = app.admit_interactive_task(
        f"request-{function.value}", "customers", 1.0, function, parameters,
    )

    assert decision.outcome.complete
    assert decision.outcome.terminal_reason is InteractiveTerminalReason.COMPLETE
    assert tuple(item.operation_id for item in decision.outcome.results)[0] == "query:customers"
    task_result = decision.outcome.results[1]
    assert task_result.operation_id == (
        f"task:{function.value}:{app.task_definition_versions[function.value]}:pack:0"
    )
    assert tuple(dict(record) for record in task_result.value) == (
        {"id": "customer-1", field: expected},
    )
    assert app.read_adapter.read_count == 1
    assert app.model_provider.calls == 1
    assert app.effect_adapter.mutation_count == 0
