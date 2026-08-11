"""Immutable provider-neutral models for supported AI task functions."""

from dataclasses import dataclass, field
from enum import StrEnum

from ai_retrieval.domain.immutable import FrozenMapping
from ai_retrieval.domain.work import ModelWork


class TaskFunction(StrEnum):
    """Task functions supported by the provider-neutral task layer."""

    CLASSIFY = "ai_classify"
    SUMMARIZE = "ai_summarize"


class TaskFailureStage(StrEnum):
    """Stable processing stages used to classify task failures."""

    DEFINITION = "definition"
    RESOLUTION = "resolution"
    INVOCATION = "invocation"
    PACKING = "packing"
    PARSING = "parsing"
    VALIDATION = "validation"


class TaskFailureCode(StrEnum):
    """Stable machine-readable task failure codes."""

    INVALID_TASK_DEFINITION = "invalid_task_definition"
    TASK_DEFINITION_UNAVAILABLE = "task_definition_unavailable"
    TASK_DEFINITION_INCOMPATIBLE = "task_definition_incompatible"
    INVALID_TASK_INVOCATION = "invalid_task_invocation"
    OVERSIZED_ROW = "oversized_row"
    RESPONSE_TOO_LARGE = "response_too_large"
    UNSUPPORTED_REPRESENTATION = "unsupported_representation"
    MALFORMED_STRUCTURED_OUTPUT = "malformed_structured_output"
    OUTPUT_SCHEMA_VIOLATION = "output_schema_violation"
    EXACT_ROW_MAPPING_VIOLATION = "exact_row_mapping_violation"
    LABEL_NOT_ALLOWED = "label_not_allowed"
    SUMMARY_WORD_LIMIT_EXCEEDED = "summary_word_limit_exceeded"
    SUMMARY_CHARACTER_LIMIT_EXCEEDED = "summary_character_limit_exceeded"


def _require_task_function(function: object) -> None:
    if not isinstance(function, TaskFunction):
        raise ValueError("task function must be ai_classify or ai_summarize")


def _require_frozen_mapping(value: object, field_name: str) -> None:
    if not isinstance(value, FrozenMapping):
        raise TypeError(f"{field_name} must be a FrozenMapping")


@dataclass(frozen=True)
class PackingLimits:
    """Candidate upper bounds for one packed provider-neutral request."""

    max_items: int
    max_input_tokens: int
    max_output_tokens: int
    max_payload_bytes: int


@dataclass(frozen=True)
class SummaryLengthLimits:
    """Candidate source and output bounds for ``ai_summarize``."""

    max_source_characters: int
    max_words: int
    max_summary_characters: int


@dataclass(frozen=True)
class TaskDefinition:
    """Immutable candidate definition validated by the task registry."""

    function: TaskFunction
    version: str
    prompt_template: str
    parameter_contract: FrozenMapping
    payload_contract: FrozenMapping
    output_schema: FrozenMapping
    required_capabilities: frozenset[str]
    validation_rules_version: str
    packing_limits: PackingLimits
    response_size_limit: int
    label_count_limit: int | None = None
    label_character_limit: int | None = None
    summary_limits: SummaryLengthLimits | None = None

    def __post_init__(self) -> None:
        _require_task_function(self.function)
        _require_frozen_mapping(self.parameter_contract, "parameter_contract")
        _require_frozen_mapping(self.payload_contract, "payload_contract")
        _require_frozen_mapping(self.output_schema, "output_schema")
        if not isinstance(self.required_capabilities, frozenset):
            raise TypeError("required_capabilities must be a frozenset")


@dataclass(frozen=True)
class RowInput:
    """One identified row with recursively immutable source fields."""

    identifier: str
    source_fields: FrozenMapping

    def __post_init__(self) -> None:
        _require_frozen_mapping(self.source_fields, "source_fields")
        if not self.identifier.strip():
            raise ValueError("row identifier must not be blank")


@dataclass(frozen=True)
class TaskInvocation:
    """A task selection, canonical parameters, and ordered input rows."""

    function: TaskFunction
    parameters: FrozenMapping
    rows: tuple[RowInput, ...]
    requested_definition_version: str | None = None

    def __post_init__(self) -> None:
        _require_task_function(self.function)
        _require_frozen_mapping(self.parameters, "parameters")
        if not isinstance(self.rows, tuple):
            raise TypeError("task invocation rows must be a tuple")
        if not self.rows:
            raise ValueError("task invocation requires at least one row")
        identifiers = tuple(row.identifier for row in self.rows)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("task invocation row identifiers must be unique")
        if self.requested_definition_version is not None and not self.requested_definition_version.strip():
            raise ValueError("requested task definition version must not be blank")


@dataclass(frozen=True)
class PackedRequest:
    """Canonical bytes and estimates for one contiguous input-row slice."""

    function: TaskFunction
    definition_version: str
    pack_index: int
    input_ids: tuple[str, ...]
    payload: bytes
    estimated_input_tokens: int
    estimated_output_tokens: int

    def __post_init__(self) -> None:
        _require_task_function(self.function)
        if not self.definition_version.strip():
            raise ValueError("packed request definition version must not be blank")
        if isinstance(self.pack_index, bool) or not isinstance(self.pack_index, int) or self.pack_index < 0:
            raise ValueError("packed request index must be a nonnegative integer")
        if not isinstance(self.input_ids, tuple):
            raise TypeError("packed request input identifiers must be a tuple")
        if not isinstance(self.payload, bytes):
            raise TypeError("packed request payload must be bytes")
        if not self.input_ids or any(not identifier.strip() for identifier in self.input_ids):
            raise ValueError("packed request requires non-blank input identifiers")
        if len(self.input_ids) != len(set(self.input_ids)):
            raise ValueError("packed request input identifiers must be unique")
        if not self.payload:
            raise ValueError("packed request payload must not be empty")
        estimates = (self.estimated_input_tokens, self.estimated_output_tokens)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in estimates):
            raise ValueError("packed request token estimates must be nonnegative integers")

@dataclass(frozen=True)
class PreparedTask:
    """A resolved definition and its exact ordered packed execution plan."""

    definition: TaskDefinition
    invocation: TaskInvocation
    packs: tuple[PackedRequest, ...]
    model_work: tuple[ModelWork, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.packs, tuple) or not isinstance(self.model_work, tuple):
            raise TypeError("prepared task packs and ModelWork values must be tuples")
        if self.definition.function is not self.invocation.function:
            raise ValueError("prepared task definition and invocation functions must match")
        if not self.packs:
            raise ValueError("prepared task requires at least one packed request")
        if len(self.packs) != len(self.model_work):
            raise ValueError("prepared task requires one ModelWork per packed request")
        expected_ids = tuple(row.identifier for row in self.invocation.rows)
        packed_ids = tuple(identifier for pack in self.packs for identifier in pack.input_ids)
        if packed_ids != expected_ids:
            raise ValueError("prepared task packs must exactly preserve invocation row order")
        for index, (pack, work) in enumerate(zip(self.packs, self.model_work, strict=True)):
            if pack.function is not self.definition.function:
                raise ValueError("prepared task pack function must match its definition")
            if pack.definition_version != self.definition.version:
                raise ValueError("prepared task pack version must match its definition")
            if pack.pack_index != index:
                raise ValueError("prepared task pack indexes must be contiguous from zero")
            if work.task_type != self.definition.function.value or work.input_ids != pack.input_ids:
                raise ValueError("prepared task ModelWork must match its packed request")


@dataclass(frozen=True)
class TaskFailure:
    """A stable task failure containing only bounded, redacted metadata."""

    stage: TaskFailureStage
    code: TaskFailureCode
    failed_rule_ids: tuple[str, ...] = ()
    details: FrozenMapping = field(default_factory=lambda: FrozenMapping(()))

    def __post_init__(self) -> None:
        if not isinstance(self.stage, TaskFailureStage) or not isinstance(self.code, TaskFailureCode):
            raise ValueError("task failure stage and code must be stable task values")
        if not isinstance(self.failed_rule_ids, tuple):
            raise TypeError("task failure rule identifiers must be a tuple")
        _require_frozen_mapping(self.details, "details")
        if any(not rule_id.strip() for rule_id in self.failed_rule_ids):
            raise ValueError("task failure rule identifiers must not be blank")
        if len(self.failed_rule_ids) != len(set(self.failed_rule_ids)):
            raise ValueError("task failure rule identifiers must be unique")


@dataclass(frozen=True)
class TaskOutputFailure(TaskFailure):
    """Typed parser/validation failure retained for downstream compatibility."""
