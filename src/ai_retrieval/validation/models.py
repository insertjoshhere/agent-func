"""Immutable deterministic validation rules, outcomes, and metadata."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ai_retrieval.domain.immutable import FrozenMapping


class FieldType(StrEnum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"


@dataclass(frozen=True)
class FieldRule:
    rule_id: str
    field: str
    value_type: FieldType
    required: bool = True
    nullable: bool = False
    allowed_values: frozenset[Any] | None = None
    minimum: int | float | None = None
    maximum: int | float | None = None

    def __post_init__(self) -> None:
        if not self.rule_id.strip() or not self.field.strip():
            raise ValueError("validation rule and field identifiers must not be blank")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("validation rule minimum must not exceed maximum")


@dataclass(frozen=True)
class ValidationRules:
    version: str
    identifier_field: str
    field_rules: tuple[FieldRule, ...]
    allow_additional_fields: bool = False

    def __post_init__(self) -> None:
        if not self.version.strip() or not self.identifier_field.strip():
            raise ValueError("validation version and identifier field must not be blank")
        rule_ids = tuple(rule.rule_id for rule in self.field_rules)
        fields = tuple(rule.field for rule in self.field_rules)
        if len(rule_ids) != len(set(rule_ids)) or len(fields) != len(set(fields)):
            raise ValueError("validation rule identifiers and fields must be unique")


class ValidationStatus(StrEnum):
    ACCEPTED = "accepted"
    VALIDATION_FAILED = "validation-failed"


@dataclass(frozen=True)
class ValidationOutcome:
    accepted: bool
    status: ValidationStatus
    failed_rule_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]
    input_identifier_digest: str
    output_identifier_digest: str
    rules_version: str


@dataclass(frozen=True)
class ValidationMetadata:
    rules_version: str
    outcome: ValidationStatus
    failed_rule_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]
    input_identifier_digest: str
    output_identifier_digest: str


@dataclass(frozen=True)
class ValidationResult:
    outcome: ValidationOutcome
    accepted_records: tuple[FrozenMapping, ...] = ()
    metadata_recorded: bool = False

    def __post_init__(self) -> None:
        if not self.outcome.accepted and self.accepted_records:
            raise ValueError("validation-failed results must withhold output")
