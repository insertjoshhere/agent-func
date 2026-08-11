"""Focused examples for canonical serialization and bounded task packing."""

import json
from dataclasses import replace

import pytest

from ai_retrieval.domain.immutable import freeze
from ai_retrieval.tasks import (
    CanonicalSerializationError,
    DeterministicTaskPacker,
    PackingLimits,
    RowInput,
    TaskFailure,
    TaskFailureCode,
    TaskFunction,
    TaskInvocation,
    canonical_payload_bytes,
)
from test_task_registry import candidate, execution_context


class ByteEstimator:
    def estimate_input(self, payload: bytes) -> int:
        return len(payload)

    def estimate_output(self, definition, row_count: int) -> int:
        return row_count * 3


class RecordingStore:
    def __init__(self) -> None:
        self.calls: list[tuple[bytes, str]] = []

    def put(self, payload: bytes, content_hash: str) -> str:
        self.calls.append((payload, content_hash))
        return f"payload://{content_hash}"


def resolved(rows, *, limits=PackingLimits(2, 10_000, 6, 10_000)):
    from ai_retrieval.tasks import ResolvedTask

    definition = replace(candidate(), version="definition-1", packing_limits=limits)
    invocation = TaskInvocation(
        TaskFunction.CLASSIFY,
        freeze({"labels": ["β", "alpha"]}),
        tuple(rows),
    )
    return ResolvedTask(definition, invocation)


def test_canonical_json_is_utf8_sorted_compact_and_preserves_array_order() -> None:
    value = freeze({"z": ["β", "a"], "a": {"n": 1}})

    encoded = canonical_payload_bytes(value)

    assert encoded == b'{"a":{"n":1},"z":["\xce\xb2","a"]}'
    assert json.loads(encoded) == {"a": {"n": 1}, "z": ["β", "a"]}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), object(), frozenset({"x"})])
def test_canonical_json_rejects_non_finite_or_unsupported_values(value) -> None:
    with pytest.raises(CanonicalSerializationError):
        canonical_payload_bytes(value)


def test_greedy_packing_is_contiguous_deterministic_bounded_and_faithful() -> None:
    rows = [
        RowInput("r1", freeze({"text": "é", "nested": {"b": 2, "a": 1}})),
        RowInput("r2", freeze({"text": "second"})),
        RowInput("r3", freeze({"text": "third"})),
    ]
    first_store = RecordingStore()
    second_store = RecordingStore()

    first = DeterministicTaskPacker(ByteEstimator(), first_store).prepare(
        resolved(rows), execution_context()
    )
    second = DeterministicTaskPacker(ByteEstimator(), second_store).prepare(
        resolved(rows), execution_context()
    )

    assert not isinstance(first, TaskFailure)
    assert not isinstance(second, TaskFailure)
    assert tuple(pack.input_ids for pack in first.packs) == (("r1", "r2"), ("r3",))
    assert tuple(pack.payload for pack in first.packs) == tuple(pack.payload for pack in second.packs)
    assert all(len(pack.input_ids) <= 2 for pack in first.packs)
    assert all(pack.estimated_output_tokens <= 6 for pack in first.packs)
    decoded = [json.loads(pack.payload) for pack in first.packs]
    assert [row["id"] for payload in decoded for row in payload["rows"]] == ["r1", "r2", "r3"]
    assert decoded[0]["parameters"]["labels"] == ["β", "alpha"]
    assert decoded[0]["rows"][0]["source"] == {"nested": {"a": 1, "b": 2}, "text": "é"}


def test_oversized_later_row_rejects_before_any_payload_is_persisted() -> None:
    small = RowInput("small", freeze({"text": "ok"}))
    large = RowInput("large", freeze({"text": "x" * 500}))
    store = RecordingStore()
    definition = resolved((small, large))
    one_row_size = len(canonical_payload_bytes({
        "task": definition.definition.function.value,
        "definition_version": definition.definition.version,
        "instructions": definition.definition.prompt_template,
        "parameters": definition.invocation.parameters,
        "rows": [{"id": small.identifier, "source": small.source_fields}],
        "output_schema": definition.definition.output_schema,
    }))
    from ai_retrieval.tasks import ResolvedTask

    bounded = ResolvedTask(
        replace(definition.definition, packing_limits=PackingLimits(1, 10_000, 3, one_row_size)),
        definition.invocation,
    )

    failure = DeterministicTaskPacker(ByteEstimator(), store).prepare(
        bounded, execution_context()
    )

    assert isinstance(failure, TaskFailure)
    assert failure.code is TaskFailureCode.OVERSIZED_ROW
    assert failure.details["row_id"] == "large"
    assert store.calls == []


def test_model_work_carries_payload_reference_capabilities_version_and_estimates() -> None:
    store = RecordingStore()
    prepared = DeterministicTaskPacker(ByteEstimator(), store).prepare(
        resolved((RowInput("row-1", freeze({"text": "value"})),)),
        execution_context(),
    )

    assert not isinstance(prepared, TaskFailure)
    pack = prepared.packs[0]
    work = prepared.model_work[0]
    assert work.task_type == TaskFunction.CLASSIFY.value
    assert work.payload_reference == f"payload://{store.calls[0][1]}"
    assert work.input_ids == pack.input_ids == ("row-1",)
    assert work.required_capabilities == prepared.definition.required_capabilities
    assert work.task_definition_version == prepared.definition.version
    assert work.estimated_input_tokens == pack.estimated_input_tokens
    assert work.estimated_output_tokens == pack.estimated_output_tokens


class RecordingEstimator(ByteEstimator):
    def __init__(self) -> None:
        self.calls = 0

    def estimate_input(self, payload: bytes) -> int:
        self.calls += 1
        return super().estimate_input(payload)

    def estimate_output(self, definition, row_count: int) -> int:
        self.calls += 1
        return super().estimate_output(definition, row_count)


def task_runtime(function: TaskFunction):
    from ai_retrieval.tasks import TaskDefinitionRegistry, TaskRuntime, task_output_schema
    from test_task_registry import rules_registry

    definition = replace(
        candidate(function),
        output_schema=task_output_schema(function),
        packing_limits=PackingLimits(10, 10_000, 10_000, 10_000),
    )
    registry = TaskDefinitionRegistry(rules_registry())
    bound = registry.register(definition)
    estimator = RecordingEstimator()
    store = RecordingStore()
    runtime = TaskRuntime(registry, DeterministicTaskPacker(estimator, store))
    context = execution_context({function.value: bound.version})
    return runtime, context, bound, estimator, store


def test_classify_runtime_preserves_ordered_labels_and_exact_payload_contract() -> None:
    runtime, context, bound, estimator, store = task_runtime(TaskFunction.CLASSIFY)
    invocation = TaskInvocation(
        TaskFunction.CLASSIFY,
        freeze({"labels": ["βeta", "alpha"]}),
        (RowInput("row-1", freeze({"text": "source"})),),
    )

    prepared = runtime.prepare(invocation, context)

    assert not isinstance(prepared, TaskFailure)
    payload = json.loads(prepared.packs[0].payload)
    assert set(payload) == {
        "task", "definition_version", "instructions", "parameters", "rows", "output_schema"
    }
    assert payload["task"] == TaskFunction.CLASSIFY.value
    assert payload["definition_version"] == bound.version
    assert payload["instructions"] == bound.prompt_template
    assert payload["parameters"] == {"labels": ["βeta", "alpha"]}
    assert payload["rows"] == [{"id": "row-1", "source": {"text": "source"}}]
    assert payload["output_schema"] == {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"id": {"type": "string"}, "label": {"type": "string"}},
            "required": ["id", "label"],
            "additionalProperties": False,
        },
    }
    assert prepared.model_work[0].required_capabilities == bound.required_capabilities
    assert estimator.calls > 0
    assert len(store.calls) == 1


@pytest.mark.parametrize(
    "labels,rule_id",
    [
        ([], "task_invocation.classify.labels.nonempty"),
        (["ok", " "], "task_invocation.classify.labels.nonblank"),
        (["same", "same"], "task_invocation.classify.labels.unique"),
        (["a", "b", "c", "d", "e"], "task_invocation.classify.labels.count_limit"),
        (["x" * 21], "task_invocation.classify.labels.character_limit"),
    ],
)
def test_invalid_classify_labels_fail_before_estimator_and_store(labels, rule_id) -> None:
    runtime, context, _, estimator, store = task_runtime(TaskFunction.CLASSIFY)
    invocation = TaskInvocation(
        TaskFunction.CLASSIFY,
        freeze({"labels": labels}),
        (RowInput("row-1", freeze({"text": "source"})),),
    )

    failure = runtime.prepare(invocation, context)

    assert isinstance(failure, TaskFailure)
    assert failure.code is TaskFailureCode.INVALID_TASK_INVOCATION
    assert rule_id in failure.failed_rule_ids
    assert estimator.calls == 0
    assert store.calls == []


def test_summarize_runtime_preserves_max_words_text_and_exact_payload_contract() -> None:
    runtime, context, bound, _, _ = task_runtime(TaskFunction.SUMMARIZE)
    invocation = TaskInvocation(
        TaskFunction.SUMMARIZE,
        freeze({"max_words": 25}),
        (RowInput("row-1", freeze({"text": "Unicode source 📝"})),),
    )

    prepared = runtime.prepare(invocation, context)

    assert not isinstance(prepared, TaskFailure)
    payload = json.loads(prepared.packs[0].payload)
    assert payload["parameters"] == {"max_words": 25}
    assert payload["rows"] == [{"id": "row-1", "source": {"text": "Unicode source 📝"}}]
    assert payload["output_schema"]["items"] == {
        "type": "object",
        "properties": {"id": {"type": "string"}, "summary": {"type": "string"}},
        "required": ["id", "summary"],
        "additionalProperties": False,
    }
    assert prepared.definition is bound


@pytest.mark.parametrize("max_words", [True, False, 0, -1, 101, 1.5, "10"])
def test_invalid_summary_max_words_fail_before_estimator_and_store(max_words) -> None:
    runtime, context, _, estimator, store = task_runtime(TaskFunction.SUMMARIZE)
    invocation = TaskInvocation(
        TaskFunction.SUMMARIZE,
        freeze({"max_words": max_words}),
        (RowInput("row-1", freeze({"text": "source"})),),
    )

    failure = runtime.prepare(invocation, context)

    assert isinstance(failure, TaskFailure)
    assert failure.code is TaskFailureCode.INVALID_TASK_INVOCATION
    assert estimator.calls == 0
    assert store.calls == []


@pytest.mark.parametrize(
    "source,rule_id",
    [
        ({"text": "x" * 1001}, "task_invocation.summarize.source_text.character_limit"),
        ({"text": 3}, "task_invocation.summarize.source_text.string"),
        ({"body": "source"}, "task_invocation.summarize.source_text.string"),
        ({"text": "source", "extra": "no"}, "task_invocation.summarize.source_text.string"),
    ],
)
def test_invalid_summary_source_fails_before_estimator_and_store(source, rule_id) -> None:
    runtime, context, _, estimator, store = task_runtime(TaskFunction.SUMMARIZE)
    invocation = TaskInvocation(
        TaskFunction.SUMMARIZE,
        freeze({"max_words": 10}),
        (RowInput("row-1", freeze(source)),),
    )

    failure = runtime.prepare(invocation, context)

    assert isinstance(failure, TaskFailure)
    assert failure.code is TaskFailureCode.INVALID_TASK_INVOCATION
    assert rule_id in failure.failed_rule_ids
    assert estimator.calls == 0
    assert store.calls == []


# Property-based coverage for the named packing and invocation design properties.
from hypothesis import given, settings, strategies as st


_UNICODE_TEXT = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)), min_size=0, max_size=12
)
_NONBLANK_UNICODE = _UNICODE_TEXT.filter(str.strip)
_FIELD_NAMES = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)), min_size=1, max_size=8
).filter(str.strip)
_JSON_VALUES = st.recursive(
    st.one_of(
        st.none(), st.booleans(), st.integers(min_value=-10_000, max_value=10_000),
        st.floats(allow_nan=False, allow_infinity=False, width=32), _UNICODE_TEXT,
    ),
    lambda children: st.one_of(
        st.lists(children, max_size=3),
        st.dictionaries(_FIELD_NAMES, children, max_size=3),
    ),
    max_leaves=8,
)
_SOURCE_FIELDS = st.dictionaries(_FIELD_NAMES, _JSON_VALUES, min_size=1, max_size=4)


class ScaledEstimator:
    def __init__(self, input_divisor: int, output_per_row: int) -> None:
        self.input_divisor = input_divisor
        self.output_per_row = output_per_row

    def estimate_input(self, payload: bytes) -> int:
        return (len(payload) + self.input_divisor - 1) // self.input_divisor

    def estimate_output(self, definition, row_count: int) -> int:
        return row_count * self.output_per_row


def _canonical_pack_payload(definition, parameters, rows) -> bytes:
    return canonical_payload_bytes({
        "task": definition.function.value,
        "definition_version": definition.version,
        "instructions": definition.prompt_template,
        "parameters": parameters,
        "rows": [{"id": row.identifier, "source": row.source_fields} for row in rows],
        "output_schema": definition.output_schema,
    })


def _fits(limits, estimator, definition, payload, row_count) -> bool:
    return (
        row_count <= limits.max_items
        and estimator.estimate_input(payload) <= limits.max_input_tokens
        and estimator.estimate_output(definition, row_count) <= limits.max_output_tokens
        and len(payload) <= limits.max_payload_bytes
    )


def _expected_partition(resolution, estimator):
    expected = []
    current = []
    for row in resolution.invocation.rows:
        proposed = current + [row]
        payload = _canonical_pack_payload(
            resolution.definition, resolution.invocation.parameters, proposed
        )
        if _fits(
            resolution.definition.packing_limits, estimator, resolution.definition,
            payload, len(proposed),
        ):
            current = proposed
        else:
            expected.append(tuple(current))
            current = [row]
    expected.append(tuple(current))
    return tuple(expected)


@st.composite
def bounded_packing_cases(draw):
    from ai_retrieval.tasks import ResolvedTask

    sources = draw(st.lists(_SOURCE_FIELDS, min_size=1, max_size=6))
    identifiers = draw(
        st.lists(_NONBLANK_UNICODE, min_size=len(sources), max_size=len(sources), unique=True)
    )
    labels = draw(
        st.lists(_NONBLANK_UNICODE, min_size=1, max_size=4, unique=True)
    )
    rows = tuple(RowInput(identifier, freeze(source)) for identifier, source in zip(identifiers, sources))
    definition = replace(
        candidate(), version="definition-property-33",
        label_count_limit=4,
        label_character_limit=max(len(label) for label in labels),
        packing_limits=PackingLimits(100, 10**9, 10**9, 10**9),
    )
    parameters = freeze({"labels": labels})
    estimator = ScaledEstimator(
        draw(st.integers(min_value=1, max_value=16)),
        draw(st.integers(min_value=1, max_value=8)),
    )
    full_payload = _canonical_pack_payload(definition, parameters, rows)
    single_payloads = [
        _canonical_pack_payload(definition, parameters, (row,)) for row in rows
    ]
    boundary = draw(st.sampled_from(("items", "input", "output", "bytes")))
    max_items = len(rows)
    max_input = estimator.estimate_input(full_payload)
    max_output = estimator.estimate_output(definition, len(rows))
    max_bytes = len(full_payload)
    if boundary == "items" and len(rows) > 1:
        max_items = draw(st.integers(min_value=1, max_value=len(rows) - 1))
    elif boundary == "input":
        max_input = draw(st.integers(
            min_value=max(estimator.estimate_input(payload) for payload in single_payloads),
            max_value=max_input,
        ))
    elif boundary == "output":
        max_output = draw(st.integers(
            min_value=estimator.output_per_row, max_value=max_output
        ))
    elif boundary == "bytes":
        max_bytes = draw(st.integers(
            min_value=max(map(len, single_payloads)), max_value=max_bytes
        ))
    definition = replace(
        definition,
        packing_limits=PackingLimits(max_items, max_input, max_output, max_bytes),
    )
    return ResolvedTask(
        definition, TaskInvocation(TaskFunction.CLASSIFY, parameters, rows)
    ), estimator


@settings(max_examples=100, deadline=None)
@given(case=bounded_packing_cases())
def test_property_33_deterministic_bounded_ordered_packing(case) -> None:
    """**Validates: Requirements 15.1, 15.2, 15.3, 15.4**"""
    resolution, estimator = case
    first_store, second_store = RecordingStore(), RecordingStore()
    first = DeterministicTaskPacker(estimator, first_store).prepare(
        resolution, execution_context()
    )
    second = DeterministicTaskPacker(estimator, second_store).prepare(
        resolution, execution_context()
    )

    assert not isinstance(first, TaskFailure)
    assert not isinstance(second, TaskFailure)
    expected = _expected_partition(resolution, estimator)
    assert tuple(pack.input_ids for pack in first.packs) == tuple(
        tuple(row.identifier for row in rows) for rows in expected
    )
    assert tuple(pack.payload for pack in first.packs) == tuple(
        pack.payload for pack in second.packs
    )
    assert tuple(pack.payload for pack in first.packs) == tuple(
        _canonical_pack_payload(resolution.definition, resolution.invocation.parameters, rows)
        for rows in expected
    )
    decoded_rows = [
        row for pack in first.packs for row in json.loads(pack.payload)["rows"]
    ]
    assert [row["id"] for row in decoded_rows] == [
        row.identifier for row in resolution.invocation.rows
    ]
    assert [row["source"] for row in decoded_rows] == [
        json.loads(canonical_payload_bytes(row.source_fields))
        for row in resolution.invocation.rows
    ]
    limits = resolution.definition.packing_limits
    assert all(len(pack.input_ids) <= limits.max_items for pack in first.packs)
    assert all(pack.estimated_input_tokens <= limits.max_input_tokens for pack in first.packs)
    assert all(pack.estimated_output_tokens <= limits.max_output_tokens for pack in first.packs)
    assert all(len(pack.payload) <= limits.max_payload_bytes for pack in first.packs)


@st.composite
def oversized_row_cases(draw):
    from ai_retrieval.tasks import ResolvedTask

    row = RowInput(draw(_NONBLANK_UNICODE), freeze(draw(_SOURCE_FIELDS)))
    estimator = ScaledEstimator(
        draw(st.integers(min_value=1, max_value=16)),
        draw(st.integers(min_value=2, max_value=8)),
    )
    definition = replace(
        candidate(), version="definition-property-34",
        packing_limits=PackingLimits(1, 10**9, 10**9, 10**9),
    )
    invocation = TaskInvocation(
        TaskFunction.CLASSIFY, freeze({"labels": ["yes", "no"]}), (row,)
    )
    payload = _canonical_pack_payload(definition, invocation.parameters, (row,))
    measures = {
        "input": estimator.estimate_input(payload),
        "output": estimator.estimate_output(definition, 1),
        "bytes": len(payload),
    }
    boundary = draw(st.sampled_from(tuple(measures)))
    limits = PackingLimits(1, measures["input"], measures["output"], measures["bytes"])
    values = list(limits.__dict__.values())
    index = {"input": 1, "output": 2, "bytes": 3}[boundary]
    values[index] = measures[boundary] - 1
    definition = replace(definition, packing_limits=PackingLimits(*values))
    return ResolvedTask(definition, invocation), estimator


@settings(max_examples=100, deadline=None)
@given(case=oversized_row_cases())
def test_property_34_oversized_rows_fail_before_model_effects(case) -> None:
    """**Validates: Requirements 15.5**"""
    resolution, estimator = case
    store = RecordingStore()

    result = DeterministicTaskPacker(estimator, store).prepare(
        resolution, execution_context()
    )

    assert isinstance(result, TaskFailure)
    assert result.code is TaskFailureCode.OVERSIZED_ROW
    assert result.details["row_id"] == resolution.invocation.rows[0].identifier
    assert store.calls == []


@settings(max_examples=100, deadline=None)
@given(case=bounded_packing_cases())
def test_property_35_packed_requests_map_faithfully_to_model_work_and_budgets(case) -> None:
    """**Validates: Requirements 15.6, 15.7**"""
    from ai_retrieval.domain.budget import Usage
    from test_interactive_integration import (
        RecordingRouter, execution_context as planning_context, planner,
    )

    resolution, estimator = case
    store = RecordingStore()
    prepared = DeterministicTaskPacker(estimator, store).prepare(
        resolution, execution_context()
    )
    assert not isinstance(prepared, TaskFailure)
    router = RecordingRouter()
    routed = planner(router)

    for pack, work, stored in zip(
        prepared.packs, prepared.model_work, store.calls, strict=True
    ):
        operation_id = f"model:{pack.pack_index}:{work.task_type}"
        plan = routed.plan(operation_id, work, planning_context(), 12)
        assert work.task_type == pack.function.value
        assert work.payload_reference == f"payload://{stored[1]}"
        assert stored[0] == pack.payload
        assert work.input_ids == pack.input_ids
        assert work.required_capabilities == prepared.definition.required_capabilities
        assert work.task_definition_version == pack.definition_version
        assert work.estimated_input_tokens == pack.estimated_input_tokens
        assert work.estimated_output_tokens == pack.estimated_output_tokens
        assert plan.operation.operation_id == operation_id
        assert plan.operation.work is work
        assert plan.operation.estimated_total_tokens == (
            pack.estimated_input_tokens + pack.estimated_output_tokens
        )
        assert plan.estimate == Usage(
            3, pack.estimated_input_tokens, pack.estimated_output_tokens
        )


def _runtime_with_definition(definition):
    from ai_retrieval.tasks import TaskDefinitionRegistry, TaskRuntime
    from test_task_registry import rules_registry

    registry = TaskDefinitionRegistry(rules_registry())
    bound = registry.register(definition)
    estimator, store = RecordingEstimator(), RecordingStore()
    runtime = TaskRuntime(registry, DeterministicTaskPacker(estimator, store))
    return runtime, execution_context({bound.function.value: bound.version}), bound, estimator, store


@settings(max_examples=100, deadline=None)
@given(
    labels=st.lists(_UNICODE_TEXT, max_size=6),
    count_limit=st.integers(min_value=1, max_value=4),
    character_limit=st.integers(min_value=1, max_value=8),
    source=_SOURCE_FIELDS,
)
def test_property_36_classification_parameter_and_payload_contract(
    labels, count_limit, character_limit, source
) -> None:
    """**Validates: Requirements 16.1, 16.2, 16.3**"""
    definition = replace(
        candidate(), label_count_limit=count_limit,
        label_character_limit=character_limit,
        packing_limits=PackingLimits(10, 100_000, 100_000, 100_000),
    )
    runtime, context, bound, estimator, store = _runtime_with_definition(definition)
    invocation = TaskInvocation(
        TaskFunction.CLASSIFY, freeze({"labels": labels}),
        (RowInput("row-λ", freeze(source)),),
    )
    result = runtime.prepare(invocation, context)
    valid = (
        bool(labels) and all(label.strip() for label in labels)
        and len(labels) == len(set(labels)) and len(labels) <= count_limit
        and all(len(label) <= character_limit for label in labels)
    )

    if not valid:
        assert isinstance(result, TaskFailure)
        assert estimator.calls == 0
        assert store.calls == []
        return
    assert not isinstance(result, TaskFailure)
    payload = json.loads(result.packs[0].payload)
    assert payload["parameters"] == {"labels": labels}
    assert payload["definition_version"] == bound.version
    assert payload["rows"] == [{
        "id": "row-λ", "source": json.loads(canonical_payload_bytes(freeze(source)))
    }]
    assert payload["output_schema"]["items"] == {
        "type": "object",
        "properties": {"id": {"type": "string"}, "label": {"type": "string"}},
        "required": ["id", "label"],
        "additionalProperties": False,
    }


@st.composite
def summary_invocation_cases(draw):
    source_limit = draw(st.integers(min_value=1, max_value=20))
    word_limit = draw(st.integers(min_value=1, max_value=20))
    texts = draw(st.lists(_UNICODE_TEXT, min_size=1, max_size=3))
    max_words = draw(st.one_of(
        st.booleans(), st.integers(min_value=-2, max_value=word_limit + 3),
        st.floats(allow_nan=False, allow_infinity=False), st.text(max_size=4),
    ))
    return source_limit, word_limit, texts, max_words


@settings(max_examples=100, deadline=None)
@given(case=summary_invocation_cases())
def test_property_38_summary_invocation_and_payload_bounds(case) -> None:
    """**Validates: Requirements 17.1, 17.2, 17.3, 17.4**"""
    from ai_retrieval.tasks import SummaryLengthLimits

    source_limit, word_limit, texts, max_words = case
    definition = replace(
        candidate(TaskFunction.SUMMARIZE),
        summary_limits=SummaryLengthLimits(source_limit, word_limit, 100),
        packing_limits=PackingLimits(10, 100_000, 100_000, 100_000),
    )
    runtime, context, bound, estimator, store = _runtime_with_definition(definition)
    rows = tuple(
        RowInput(f"row-{index}", freeze({"text": text}))
        for index, text in enumerate(texts)
    )
    invocation = TaskInvocation(
        TaskFunction.SUMMARIZE, freeze({"max_words": max_words}), rows
    )
    result = runtime.prepare(invocation, context)
    valid = (
        isinstance(max_words, int) and not isinstance(max_words, bool)
        and 0 < max_words <= word_limit
        and all(len(text) <= source_limit for text in texts)
    )

    if not valid:
        assert isinstance(result, TaskFailure)
        assert estimator.calls == 0
        assert store.calls == []
        return
    assert not isinstance(result, TaskFailure)
    payload = json.loads(result.packs[0].payload)
    assert payload["parameters"] == {"max_words": max_words}
    assert payload["definition_version"] == bound.version
    assert payload["rows"] == [
        {"id": f"row-{index}", "source": {"text": text}}
        for index, text in enumerate(texts)
    ]
    assert payload["output_schema"]["items"] == {
        "type": "object",
        "properties": {"id": {"type": "string"}, "summary": {"type": "string"}},
        "required": ["id", "summary"],
        "additionalProperties": False,
    }
