"""Focused tests for strict provider-neutral structured-output parsing."""

from dataclasses import FrozenInstanceError
import json

import pytest
from hypothesis import given, settings, strategies as st

from ai_retrieval.domain.immutable import FrozenMapping, freeze
from ai_retrieval.tasks import (
    PackingLimits,
    StructuredOutputParser,
    TaskDefinition,
    TaskFailureCode,
    TaskFunction,
    TaskOutputFailure,
)


def definition(function: TaskFunction, response_size_limit: int = 4096) -> TaskDefinition:
    output_field = "label" if function is TaskFunction.CLASSIFY else "summary"
    return TaskDefinition(
        function=function,
        version="definition-1",
        prompt_template="Process rows",
        parameter_contract=freeze({"type": "object"}),
        payload_contract=freeze({"type": "object"}),
        output_schema=freeze({
            "type": "array",
            "items": {"required": ["id", output_field], "additionalProperties": False},
        }),
        required_capabilities=frozenset({function.value}),
        validation_rules_version="rules-1",
        packing_limits=PackingLimits(2, 100, 50, 4096),
        response_size_limit=response_size_limit,
    )


@pytest.mark.parametrize("function,field,value", [
    (TaskFunction.CLASSIFY, "label", "approved"),
    (TaskFunction.SUMMARIZE, "summary", "A concise summary."),
])
def test_equivalent_object_text_bytes_and_envelope_are_equal(function, field, value) -> None:
    records = [{"id": "row-1", field: value}]
    document = json.dumps(records, ensure_ascii=False)
    parser = StructuredOutputParser({"provider": lambda value: value["result"]})

    expected = parser.parse(records, definition(function))

    assert parser.parse(document, definition(function)) == expected
    assert parser.parse(document.encode("utf-8"), definition(function)) == expected
    assert parser.parse({"result": records}, definition(function), envelope="provider") == expected
    assert isinstance(expected, tuple)
    assert isinstance(expected[0], FrozenMapping)


def test_response_size_is_checked_before_utf8_or_json_decode() -> None:
    parser = StructuredOutputParser()
    oversized_invalid_utf8 = b"\xff" * 9

    failure = parser.parse(oversized_invalid_utf8, definition(TaskFunction.CLASSIFY, 8))

    assert isinstance(failure, TaskOutputFailure)
    assert failure.code is TaskFailureCode.RESPONSE_TOO_LARGE
    assert failure.details["response_size"] == 9
    assert failure.details["response_size_limit"] == 8


@pytest.mark.parametrize("response,reason", [
    (b"\xff", "invalid_utf8"),
    ('[{"id":"row-1","label":}]', "invalid_json"),
    ('[{"id":"row-1","id":"row-2","label":"yes"}]', "duplicate_object_key"),
    ('[{"id":"row-1","label":"yes","score":NaN}]', "non_finite_number"),
    ([{"id": "row-1", "label": float("inf")}], "non_finite_number"),
])
def test_malformed_documents_duplicate_keys_and_non_finite_values_are_rejected(response, reason) -> None:
    failure = StructuredOutputParser().parse(response, definition(TaskFunction.CLASSIFY))

    assert isinstance(failure, TaskOutputFailure)
    assert failure.code is TaskFailureCode.MALFORMED_STRUCTURED_OUTPUT
    assert failure.details["reason"] == reason


@pytest.mark.parametrize("response", [42, object(), bytearray(b"[]"), {"id": "row-1"}])
def test_unsupported_representations_and_non_array_roots_fail_closed(response) -> None:
    failure = StructuredOutputParser().parse(response, definition(TaskFunction.CLASSIFY))

    assert isinstance(failure, TaskOutputFailure)
    assert failure.code in {
        TaskFailureCode.UNSUPPORTED_REPRESENTATION,
        TaskFailureCode.OUTPUT_SCHEMA_VIOLATION,
    }


def test_only_allowlisted_envelopes_run_and_extractor_failures_are_redacted() -> None:
    secret = "sensitive-provider-content"
    parser = StructuredOutputParser({"configured": lambda value: value["missing"]})

    unsupported = parser.parse({"result": secret}, definition(TaskFunction.CLASSIFY), envelope="other")
    malformed = parser.parse({"result": secret}, definition(TaskFunction.CLASSIFY), envelope="configured")

    assert unsupported.code is TaskFailureCode.UNSUPPORTED_REPRESENTATION
    assert malformed.code is TaskFailureCode.MALFORMED_STRUCTURED_OUTPUT
    assert secret not in repr(unsupported)
    assert secret not in repr(malformed)


@pytest.mark.parametrize("function,response", [
    (TaskFunction.CLASSIFY, [{"id": "row-1", "label": "yes", "extra": "no"}]),
    (TaskFunction.CLASSIFY, [{"id": "row-1"}]),
    (TaskFunction.CLASSIFY, [{"id": "row-1", "label": 1}]),
    (TaskFunction.SUMMARIZE, [{"id": "row-1", "label": "wrong field"}]),
    (TaskFunction.SUMMARIZE, ["not an object"]),
])
def test_exact_task_schema_is_enforced(function, response) -> None:
    failure = StructuredOutputParser().parse(response, definition(function))

    assert isinstance(failure, TaskOutputFailure)
    assert failure.code is TaskFailureCode.OUTPUT_SCHEMA_VIOLATION
    assert failure.failed_rule_ids == ("structured_output.schema",)


def test_records_are_recursively_immutable_and_input_mutation_does_not_change_them() -> None:
    response = [{"id": "row-1", "label": "yes"}]
    parsed = StructuredOutputParser().parse(response, definition(TaskFunction.CLASSIFY))
    assert isinstance(parsed, tuple)

    response[0]["label"] = "no"

    assert parsed[0]["label"] == "yes"
    with pytest.raises(FrozenInstanceError):
        parsed[0]._items = ()


def test_failures_expose_only_bounded_structural_metadata() -> None:
    secret = "never-return-this-provider-payload"
    failure = StructuredOutputParser().parse(
        [{"id": "row-1", "label": secret, "unexpected": secret}],
        definition(TaskFunction.CLASSIFY),
    )

    assert isinstance(failure, TaskOutputFailure)
    assert secret not in repr(failure)
    assert set(failure.details) <= {"representation", "reason", "record_index"}


_SCHEMA_TEXT = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)), min_size=0, max_size=24
)


@st.composite
def valid_provider_records(draw):
    function = draw(st.sampled_from(tuple(TaskFunction)))
    field = "label" if function is TaskFunction.CLASSIFY else "summary"
    identifiers = [f"row-{index}" for index in range(draw(st.integers(0, 5)))]
    values = draw(st.lists(_SCHEMA_TEXT, min_size=len(identifiers), max_size=len(identifiers)))
    return function, [{"id": identifier, field: value} for identifier, value in zip(identifiers, values)]


@settings(max_examples=100, deadline=None)
@given(case=valid_provider_records())
def test_property_40_provider_representation_parsing_confluence(case) -> None:
    """**Validates: Requirements 18.1, 18.2**"""
    function, records = case
    task_definition = definition(function)
    document = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    parser = StructuredOutputParser({"provider": lambda response: response["result"]})

    parsed = parser.parse(records, task_definition)
    equivalents = (
        parser.parse(document, task_definition),
        parser.parse(document.encode("utf-8"), task_definition),
        parser.parse({"result": records}, task_definition, envelope="provider"),
    )

    assert isinstance(parsed, tuple)
    assert all(result == parsed for result in equivalents)
    assert all(isinstance(record, FrozenMapping) for record in parsed)


@st.composite
def invalid_provider_responses(draw):
    suffix = draw(st.text(alphabet="0123456789abcdef", min_size=12, max_size=20))
    secret = f"RAW::{suffix}::PAYLOAD"
    failure_kind = draw(st.sampled_from(
        ("malformed", "oversized", "unsupported", "duplicate", "schema")
    ))
    task_definition = definition(TaskFunction.CLASSIFY)
    if failure_kind == "malformed":
        response = f'[{ {"id": "row-1", "label": secret}!r}'
    elif failure_kind == "oversized":
        response = json.dumps([{"id": "row-1", "label": secret}])
        task_definition = definition(TaskFunction.CLASSIFY, len(response.encode("utf-8")) - 1)
    elif failure_kind == "unsupported":
        response = [{"id": "row-1", "label": object(), "secret": secret}]
    elif failure_kind == "duplicate":
        response = f'[{ {"id": "row-1", "label": "yes"}!r}]'.replace(
            "'id': 'row-1'", f"'id': 'row-1', 'id': '{secret}'"
        ).replace("'", '"')
    else:
        response = [{"id": "row-1", "label": "yes", "unexpected": secret}]
    return failure_kind, response, task_definition, secret


@settings(max_examples=100, deadline=None)
@given(case=invalid_provider_responses())
def test_property_41_structured_output_parsing_fails_closed(case) -> None:
    """**Validates: Requirements 18.3**"""
    failure_kind, response, task_definition, secret = case
    parser = StructuredOutputParser()
    acceptance_calls: list[object] = []
    write_calls: list[object] = []

    first = parser.parse(response, task_definition)
    second = parser.parse(response, task_definition)
    if isinstance(first, tuple):
        acceptance_calls.append(first)
        write_calls.append(first)

    expected_codes = {
        "malformed": TaskFailureCode.MALFORMED_STRUCTURED_OUTPUT,
        "oversized": TaskFailureCode.RESPONSE_TOO_LARGE,
        "unsupported": TaskFailureCode.UNSUPPORTED_REPRESENTATION,
        "duplicate": TaskFailureCode.MALFORMED_STRUCTURED_OUTPUT,
        "schema": TaskFailureCode.OUTPUT_SCHEMA_VIOLATION,
    }
    assert isinstance(first, TaskOutputFailure)
    assert first == second
    assert first.code is expected_codes[failure_kind]
    assert secret not in repr(first)
    assert secret not in repr(first.details)
    assert acceptance_calls == []
    assert write_calls == []
