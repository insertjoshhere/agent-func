"""Execution-bound task-definition resolution before preparation effects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ai_retrieval.domain.execution import ExecutionContext
from ai_retrieval.domain.immutable import FrozenMapping, freeze
from ai_retrieval.tasks.models import (
    PreparedTask,
    RowInput,
    TaskDefinition,
    TaskFailure,
    TaskFailureCode,
    TaskFailureStage,
    TaskFunction,
    TaskInvocation,
)


class TaskDefinitionLookup(Protocol):
    """Exact immutable task-definition lookup used by the runtime."""

    def resolve(self, function: TaskFunction, version: str) -> TaskDefinition | None: ...


@dataclass(frozen=True)
class ResolvedTask:
    """An invocation paired with the exact execution-bound definition."""

    definition: TaskDefinition
    invocation: TaskInvocation

    def __post_init__(self) -> None:
        if self.definition.function is not self.invocation.function:
            raise ValueError("resolved task definition and invocation functions must match")


class TaskPreparationBuilder(Protocol):
    """Future packing seam reached only after binding resolution succeeds."""

    def prepare(
        self, resolved: ResolvedTask, context: ExecutionContext
    ) -> PreparedTask | TaskFailure: ...


class TaskRuntime:
    """Resolve immutable execution bindings before any preparation effect."""

    def __init__(
        self,
        registry: TaskDefinitionLookup,
        preparer: TaskPreparationBuilder,
    ) -> None:
        self._registry = registry
        self._preparer = preparer

    def resolve(
        self, invocation: TaskInvocation, context: ExecutionContext
    ) -> ResolvedTask | TaskFailure:
        """Resolve only the function/version fixed in the execution snapshot."""
        content = context.configuration.content
        if "task_definitions" not in content:
            return _resolution_failure(
                TaskFailureCode.TASK_DEFINITION_UNAVAILABLE,
                "task_definition.binding",
                "binding_missing",
                invocation.function,
            )
        bindings = content["task_definitions"]
        if not isinstance(bindings, FrozenMapping):
            return _resolution_failure(
                TaskFailureCode.TASK_DEFINITION_UNAVAILABLE,
                "task_definition.binding",
                "binding_malformed",
                invocation.function,
            )

        version = bindings.get(invocation.function.value)
        if not isinstance(version, str) or not version.strip():
            return _resolution_failure(
                TaskFailureCode.TASK_DEFINITION_UNAVAILABLE,
                "task_definition.binding",
                "binding_missing" if version is None else "binding_malformed",
                invocation.function,
            )

        requested = invocation.requested_definition_version
        if requested is not None and requested != version:
            return _resolution_failure(
                TaskFailureCode.TASK_DEFINITION_INCOMPATIBLE,
                "task_definition.requested_version",
                "requested_version_mismatch",
                invocation.function,
                bound_version=version,
                requested_version=requested,
            )

        definition = self._registry.resolve(invocation.function, version)
        if definition is None:
            return _resolution_failure(
                TaskFailureCode.TASK_DEFINITION_UNAVAILABLE,
                "task_definition.available",
                "definition_unavailable",
                invocation.function,
                bound_version=version,
            )
        if definition.function is not invocation.function or definition.version != version:
            return _resolution_failure(
                TaskFailureCode.TASK_DEFINITION_INCOMPATIBLE,
                "task_definition.compatible",
                "resolved_definition_mismatch",
                invocation.function,
                bound_version=version,
            )
        return ResolvedTask(definition, invocation)

    def prepare(
        self, invocation: TaskInvocation, context: ExecutionContext
    ) -> PreparedTask | TaskFailure:
        """Resolve and validate before estimator or payload-store effects."""
        resolved = self.resolve(invocation, context)
        if isinstance(resolved, TaskFailure):
            return resolved
        validated = validate_task_invocation(resolved)
        if isinstance(validated, TaskFailure):
            return validated
        return self._preparer.prepare(validated, context)


def validate_task_invocation(resolved: ResolvedTask) -> ResolvedTask | TaskFailure:
    """Return an invocation normalized to its exact task payload contract."""
    definition = resolved.definition
    invocation = resolved.invocation
    if definition.output_schema != task_output_schema(invocation.function):
        return _invocation_failure(
            ["task_invocation.output_schema.exact"],
            "incompatible_output_schema",
            invocation.function,
        )
    if invocation.function is TaskFunction.CLASSIFY:
        return _validate_classify(definition, invocation)
    return _validate_summarize(definition, invocation)


def task_output_schema(function: TaskFunction) -> FrozenMapping:
    """Return the exact provider-neutral output schema for a supported task."""
    result_field = "label" if function is TaskFunction.CLASSIFY else "summary"
    return _frozen_mapping(
        {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    result_field: {"type": "string"},
                },
                "required": ["id", result_field],
                "additionalProperties": False,
            },
        }
    )


def _validate_classify(
    definition: TaskDefinition, invocation: TaskInvocation
) -> ResolvedTask | TaskFailure:
    labels = invocation.parameters.get("labels")
    failed: list[str] = []
    reason = "invalid_labels"
    if not isinstance(labels, tuple) or not labels:
        failed.append("task_invocation.classify.labels.nonempty")
    else:
        if any(not isinstance(label, str) or not label.strip() for label in labels):
            failed.append("task_invocation.classify.labels.nonblank")
        if len(labels) != len(set(labels)):
            failed.append("task_invocation.classify.labels.unique")
        count_limit = definition.label_count_limit
        if count_limit is None or len(labels) > count_limit:
            failed.append("task_invocation.classify.labels.count_limit")
        character_limit = definition.label_character_limit
        if character_limit is None or any(
            isinstance(label, str) and len(label) > character_limit for label in labels
        ):
            failed.append("task_invocation.classify.labels.character_limit")
    if set(invocation.parameters) != {"labels"}:
        failed.append("task_invocation.classify.parameters.exact")
    if failed:
        return _invocation_failure(failed, reason, invocation.function)
    assert isinstance(labels, tuple)
    return ResolvedTask(definition, _normalized_invocation(invocation, {"labels": labels}))


def _validate_summarize(
    definition: TaskDefinition, invocation: TaskInvocation
) -> ResolvedTask | TaskFailure:
    max_words = invocation.parameters.get("max_words")
    failed: list[str] = []
    reason = "invalid_summary_invocation"
    limits = definition.summary_limits
    if isinstance(max_words, bool) or not isinstance(max_words, int) or max_words <= 0:
        failed.append("task_invocation.summarize.max_words.positive_integer")
    elif limits is None or max_words > limits.max_words:
        failed.append("task_invocation.summarize.max_words.limit")
    if set(invocation.parameters) != {"max_words"}:
        failed.append("task_invocation.summarize.parameters.exact")

    normalized_rows: list[RowInput] = []
    for row in invocation.rows:
        text = row.source_fields.get("text")
        if set(row.source_fields) != {"text"} or not isinstance(text, str):
            failed.append("task_invocation.summarize.source_text.string")
            continue
        if limits is None or len(text) > limits.max_source_characters:
            failed.append("task_invocation.summarize.source_text.character_limit")
        normalized_rows.append(RowInput(row.identifier, _frozen_mapping({"text": text})))
    if failed:
        return _invocation_failure(failed, reason, invocation.function)
    assert isinstance(max_words, int) and not isinstance(max_words, bool)
    normalized = TaskInvocation(
        invocation.function,
        _frozen_mapping({"max_words": max_words}),
        tuple(normalized_rows),
        invocation.requested_definition_version,
    )
    return ResolvedTask(definition, normalized)


def _normalized_invocation(
    invocation: TaskInvocation, parameters: dict[str, object]
) -> TaskInvocation:
    return TaskInvocation(
        invocation.function,
        _frozen_mapping(parameters),
        invocation.rows,
        invocation.requested_definition_version,
    )


def _frozen_mapping(value: dict[str, object]) -> FrozenMapping:
    frozen = freeze(value)
    assert isinstance(frozen, FrozenMapping)
    return frozen


def _invocation_failure(
    failed_rule_ids: list[str], reason: str, function: TaskFunction
) -> TaskFailure:
    return TaskFailure(
        TaskFailureStage.INVOCATION,
        TaskFailureCode.INVALID_TASK_INVOCATION,
        tuple(dict.fromkeys(failed_rule_ids)),
        _frozen_mapping({"function": function.value, "reason": reason}),
    )


def _resolution_failure(
    code: TaskFailureCode,
    rule_id: str,
    reason: str,
    function: TaskFunction,
    *,
    bound_version: str | None = None,
    requested_version: str | None = None,
) -> TaskFailure:
    details: dict[str, object] = {
        "function": function.value,
        "reason": reason,
    }
    if bound_version is not None:
        details["bound_version"] = bound_version
    if requested_version is not None:
        details["requested_version"] = requested_version
    immutable_details = freeze(details)
    assert isinstance(immutable_details, FrozenMapping)
    return TaskFailure(
        TaskFailureStage.RESOLUTION,
        code,
        (rule_id,),
        immutable_details,
    )
