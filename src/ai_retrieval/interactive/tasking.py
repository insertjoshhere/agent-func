"""Read-only task preparation and provider-neutral output processing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from ai_retrieval.domain.execution import ExecutionContext
from ai_retrieval.domain.immutable import FrozenMapping, freeze
from ai_retrieval.domain.work import InteractiveTaskWork
from ai_retrieval.relational.models import NormalizedResult
from ai_retrieval.tasks import (
    PackedRequest,
    PreparedTask,
    RowInput,
    StructuredOutputParser,
    TaskFailure,
    TaskFailureCode,
    TaskFailureStage,
    TaskFunction,
    TaskInvocation,
    TaskOutputFailure,
    TaskOutputValidator,
    TaskRuntime,
)
from ai_retrieval.validation.models import ValidationResult


class TaskRowAdapter(Protocol):
    """Convert an ordered normalized read result without database effects."""

    def rows(self, source: NormalizedResult, context: ExecutionContext) -> tuple[RowInput, ...] | TaskFailure: ...


@dataclass(frozen=True)
class ConfiguredTaskRowAdapter:
    """Map configured normalized-result columns to ordered immutable rows."""

    configuration_key: str = "task_input"

    def rows(
        self, source: NormalizedResult, context: ExecutionContext
    ) -> tuple[RowInput, ...] | TaskFailure:
        if not isinstance(source, NormalizedResult):
            return _row_failure("normalized_result_required")
        interactive = context.configuration.content.get("interactive")
        mapping = interactive.get(self.configuration_key) if isinstance(interactive, FrozenMapping) else None
        if not isinstance(mapping, FrozenMapping):
            return _row_failure("row_mapping_missing")
        identifier_column = mapping.get("identifier_column")
        source_fields = mapping.get("source_fields")
        if not isinstance(identifier_column, str) or not identifier_column.strip():
            return _row_failure("identifier_column_invalid")
        selected = _selected_columns(source_fields)
        if selected is None or not selected:
            return _row_failure("source_fields_invalid")

        names = tuple(column.name for column in source.columns)
        if len(names) != len(set(names)):
            return _row_failure("result_columns_duplicate")
        indexes = {name: index for index, name in enumerate(names)}
        required = (identifier_column, *(column for _, column in selected))
        missing = tuple(dict.fromkeys(column for column in required if column not in indexes))
        if missing:
            return _row_failure("configured_column_missing", missing_columns=missing)

        rows: list[RowInput] = []
        identifiers: set[str] = set()
        for row_index, values in enumerate(source.rows):
            if len(values) != len(source.columns):
                return _row_failure("row_width_mismatch", row_index=row_index)
            identifier = values[indexes[identifier_column]]
            if not isinstance(identifier, str) or not identifier.strip():
                return _row_failure("row_identifier_invalid", row_index=row_index)
            if identifier in identifiers:
                return _row_failure("row_identifier_duplicate", row_index=row_index)
            identifiers.add(identifier)
            try:
                fields = freeze({field: values[indexes[column]] for field, column in selected})
            except TypeError:
                return _row_failure("source_value_not_canonical", row_index=row_index)
            assert isinstance(fields, FrozenMapping)
            rows.append(RowInput(identifier, fields))
        if not rows:
            return _row_failure("normalized_result_empty")
        return tuple(rows)


class InteractiveTaskProcessor:
    """Compose task runtime, parser, and validator behind a read-only seam."""

    def __init__(
        self,
        runtime: TaskRuntime,
        parser: StructuredOutputParser,
        validator: TaskOutputValidator,
        row_adapter: TaskRowAdapter | None = None,
    ) -> None:
        self._runtime = runtime
        self._parser = parser
        self._validator = validator
        self._row_adapter = row_adapter or ConfiguredTaskRowAdapter()

    def prepare(
        self,
        selection: InteractiveTaskWork,
        source: NormalizedResult,
        context: ExecutionContext,
    ) -> PreparedTask | TaskFailure:
        try:
            function = TaskFunction(selection.function)
        except ValueError:
            return _task_failure("unsupported_task_function")
        rows = self._row_adapter.rows(source, context)
        if isinstance(rows, TaskFailure):
            return rows
        try:
            invocation = TaskInvocation(
                function,
                selection.parameters,
                rows,
                selection.requested_definition_version,
            )
        except (TypeError, ValueError):
            return _task_failure("task_invocation_invalid")
        return self._runtime.prepare(invocation, context)

    def parse_and_validate(
        self,
        prepared: PreparedTask,
        pack: PackedRequest,
        response: object,
        context: ExecutionContext,
    ) -> ValidationResult | TaskOutputFailure:
        parsed = self._parser.parse(response, prepared.definition)
        if isinstance(parsed, TaskOutputFailure):
            return parsed
        return self._validator.validate(prepared, pack, parsed, context)


def _selected_columns(value: object) -> tuple[tuple[str, str], ...] | None:
    if isinstance(value, FrozenMapping):
        selected = tuple(value.items())
    elif isinstance(value, tuple):
        selected = tuple((column, column) for column in value)
    else:
        return None
    if any(
        not isinstance(field, str) or not field.strip()
        or not isinstance(column, str) or not column.strip()
        for field, column in selected
    ):
        return None
    fields = tuple(field for field, _ in selected)
    return selected if len(fields) == len(set(fields)) else None


def _task_failure(reason: str) -> TaskFailure:
    return TaskFailure(
        TaskFailureStage.INVOCATION,
        TaskFailureCode.INVALID_TASK_INVOCATION,
        ("interactive_task.function",),
        _details(reason),
    )


def _row_failure(reason: str, **details: object) -> TaskFailure:
    return TaskFailure(
        TaskFailureStage.INVOCATION,
        TaskFailureCode.INVALID_TASK_INVOCATION,
        ("interactive_task.row_mapping",),
        _details(reason, **details),
    )


def _details(reason: str, **values: object) -> FrozenMapping:
    result = freeze({"reason": reason, **values})
    assert isinstance(result, FrozenMapping)
    return result


__all__ = [
    "ConfiguredTaskRowAdapter",
    "InteractiveTaskProcessor",
    "TaskRowAdapter",
]
