"""Aggregate validation and content-addressed storage for task definitions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from hashlib import sha256
import json
import math
from typing import Protocol

from ai_retrieval.domain.immutable import FrozenMapping
from ai_retrieval.tasks.models import (
    PackingLimits,
    SummaryLengthLimits,
    TaskDefinition,
    TaskFunction,
)


class ValidationRulesLookup(Protocol):
    """The rule-registry capability required during definition registration."""

    def resolve(self, version: str | None) -> object | None: ...


class TaskDefinitionValidationError(ValueError):
    """All validation failures found while registering one definition."""

    def __init__(self, failed_rule_ids: Sequence[str]) -> None:
        self.failed_rule_ids = tuple(dict.fromkeys(failed_rule_ids))
        super().__init__(
            "invalid task definition: " + ", ".join(self.failed_rule_ids)
        )


class TaskDefinitionRegistry:
    """Immutable-by-version content-addressed task-definition registry."""

    def __init__(self, validation_rules: ValidationRulesLookup) -> None:
        self._validation_rules = validation_rules
        self._definitions: dict[tuple[TaskFunction, str], TaskDefinition] = {}
        self._canonical_by_version: dict[str, bytes] = {}

    def register(self, candidate: TaskDefinition) -> TaskDefinition:
        """Validate and register a definition, returning its versioned copy."""
        failed = _validate_definition(candidate, self._validation_rules)
        canonical: bytes | None = None
        assigned_version: str | None = None
        try:
            canonical = canonical_definition_bytes(candidate)
            assigned_version = sha256(canonical).hexdigest()
        except (AttributeError, TypeError, ValueError):
            failed.append("task_definition.canonical_content")

        supplied_version = candidate.version
        if not isinstance(supplied_version, str):
            failed.append("task_definition.version")
        elif assigned_version is not None and supplied_version and supplied_version != assigned_version:
            failed.append("task_definition.version")

        if isinstance(supplied_version, str) and supplied_version:
            registered_content = self._canonical_by_version.get(supplied_version)
            if registered_content is not None and registered_content != canonical:
                failed.append("task_definition.version_immutable")

        if failed:
            raise TaskDefinitionValidationError(failed)

        assert canonical is not None and assigned_version is not None
        registered_content = self._canonical_by_version.get(assigned_version)
        if registered_content is not None and registered_content != canonical:
            raise TaskDefinitionValidationError(("task_definition.version_immutable",))

        definition = replace(candidate, version=assigned_version)
        key = (definition.function, assigned_version)
        existing = self._definitions.get(key)
        if existing is not None:
            return existing
        self._canonical_by_version[assigned_version] = canonical
        self._definitions[key] = definition
        return definition

    def resolve(self, function: TaskFunction, version: str) -> TaskDefinition | None:
        """Resolve only an exact function/version pair."""
        if not isinstance(function, TaskFunction) or not version:
            return None
        return self._definitions.get((function, version))

    @property
    def registered_definitions(self) -> tuple[TaskDefinition, ...]:
        """Return an immutable deterministic snapshot for configuration seeding."""
        return tuple(
            definition
            for _, definition in sorted(
                self._definitions.items(), key=lambda item: (item[0][0].value, item[0][1])
            )
        )


def canonical_definition_bytes(definition: TaskDefinition) -> bytes:
    """Serialize version-independent definition content as canonical UTF-8 JSON."""
    content = {
        "function": definition.function.value,
        "prompt_template": definition.prompt_template,
        "parameter_contract": _plain(definition.parameter_contract),
        "payload_contract": _plain(definition.payload_contract),
        "output_schema": _plain(definition.output_schema),
        "required_capabilities": sorted(definition.required_capabilities),
        "validation_rules_version": definition.validation_rules_version,
        "packing_limits": {
            "max_items": definition.packing_limits.max_items,
            "max_input_tokens": definition.packing_limits.max_input_tokens,
            "max_output_tokens": definition.packing_limits.max_output_tokens,
            "max_payload_bytes": definition.packing_limits.max_payload_bytes,
        },
        "response_size_limit": definition.response_size_limit,
        "label_count_limit": definition.label_count_limit,
        "label_character_limit": definition.label_character_limit,
        "summary_limits": (
            None
            if definition.summary_limits is None
            else {
                "max_source_characters": definition.summary_limits.max_source_characters,
                "max_words": definition.summary_limits.max_words,
                "max_summary_characters": definition.summary_limits.max_summary_characters,
            }
        ),
    }
    return json.dumps(
        content, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _plain(value: object) -> object:
    if isinstance(value, FrozenMapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, frozenset):
        values = [_plain(item) for item in value]
        return sorted(values, key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False))
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("canonical task definitions require finite numbers")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported canonical task-definition value: {type(value).__name__}")


def _positive_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _validate_definition(
    definition: TaskDefinition, validation_rules: ValidationRulesLookup
) -> list[str]:
    failed: list[str] = []
    if not isinstance(definition.function, TaskFunction):
        failed.append("task_definition.function")
    if not isinstance(definition.prompt_template, str) or not definition.prompt_template.strip():
        failed.append("task_definition.prompt_template")
    for name, contract in (
        ("parameter_contract", definition.parameter_contract),
        ("payload_contract", definition.payload_contract),
        ("output_schema", definition.output_schema),
    ):
        if not isinstance(contract, Mapping) or not contract:
            failed.append(f"task_definition.{name}")

    capabilities = definition.required_capabilities
    if (
        not isinstance(capabilities, frozenset)
        or not capabilities
        or any(not isinstance(value, str) or not value.strip() for value in capabilities)
    ):
        failed.append("task_definition.required_capabilities")

    rules_version = definition.validation_rules_version
    if not isinstance(rules_version, str) or not rules_version.strip():
        failed.append("task_definition.validation_rules_version")
    elif validation_rules.resolve(rules_version) is None:
        failed.append("task_definition.validation_rules_unavailable")

    limits = definition.packing_limits
    for name in ("max_items", "max_input_tokens", "max_output_tokens", "max_payload_bytes"):
        if not _positive_integer(getattr(limits, name, None)):
            failed.append(f"task_definition.packing_limits.{name}")
    if not _positive_integer(definition.response_size_limit):
        failed.append("task_definition.response_size_limit")

    if definition.function is TaskFunction.CLASSIFY:
        if not _positive_integer(definition.label_count_limit):
            failed.append("task_definition.label_count_limit")
        if not _positive_integer(definition.label_character_limit):
            failed.append("task_definition.label_character_limit")
        if definition.summary_limits is not None:
            failed.append("task_definition.summary_limits")
    elif definition.function is TaskFunction.SUMMARIZE:
        if definition.label_count_limit is not None:
            failed.append("task_definition.label_count_limit")
        if definition.label_character_limit is not None:
            failed.append("task_definition.label_character_limit")
        if definition.summary_limits is None:
            failed.append("task_definition.summary_limits")
        else:
            for name in ("max_source_characters", "max_words", "max_summary_characters"):
                if not _positive_integer(getattr(definition.summary_limits, name)):
                    failed.append(f"task_definition.summary_limits.{name}")
    return failed


def build_seeded_task_definition_registry(
    validation_rules: ValidationRulesLookup,
    validation_rules_version: str,
) -> TaskDefinitionRegistry:
    """Build a registry containing the two supported provider-neutral definitions."""
    registry = TaskDefinitionRegistry(validation_rules)
    for definition in _seed_definitions(validation_rules_version):
        registry.register(definition)
    return registry


def _seed_definitions(validation_rules_version: str) -> tuple[TaskDefinition, ...]:
    from ai_retrieval.domain.immutable import freeze

    common_payload = freeze({
        "type": "object",
        "required": ["task", "definition_version", "instructions", "parameters", "rows", "output_schema"],
    })
    classify_schema = freeze({
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"id": {"type": "string"}, "label": {"type": "string"}},
            "required": ["id", "label"],
            "additionalProperties": False,
        },
    })
    summary_schema = freeze({
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"id": {"type": "string"}, "summary": {"type": "string"}},
            "required": ["id", "summary"],
            "additionalProperties": False,
        },
    })
    classify = TaskDefinition(
        TaskFunction.CLASSIFY, "", "Classify each row using exactly one allowed label.",
        freeze({"type": "object", "required": ["labels"]}), common_payload,
        classify_schema,
        frozenset({TaskFunction.CLASSIFY.value}), validation_rules_version,
        PackingLimits(100, 16_384, 4_096, 1_048_576), 1_048_576,
        label_count_limit=100, label_character_limit=256,
    )
    summarize = TaskDefinition(
        TaskFunction.SUMMARIZE, "", "Summarize each row within the requested word limit.",
        freeze({"type": "object", "required": ["max_words"]}), common_payload,
        summary_schema,
        frozenset({TaskFunction.SUMMARIZE.value}), validation_rules_version,
        PackingLimits(100, 16_384, 8_192, 1_048_576), 1_048_576,
        summary_limits=SummaryLengthLimits(100_000, 2_000, 20_000),
    )
    return classify, summarize
