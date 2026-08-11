"""Version-bound structural, correspondence, and deterministic domain validation."""

from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
from typing import Protocol

from ai_retrieval.domain.execution import ExecutionContext
from ai_retrieval.domain.immutable import FrozenMapping, freeze
from ai_retrieval.validation.models import (
    FieldRule,
    FieldType,
    ValidationMetadata,
    ValidationOutcome,
    ValidationResult,
    ValidationRules,
    ValidationStatus,
)


class ValidationMetadataSink(Protocol):
    def record(self, metadata: ValidationMetadata) -> None: ...


class ValidationRulesRegistry:
    """Immutable-by-version rule registry used through the execution binding."""

    def __init__(self, rules: Sequence[ValidationRules] = ()) -> None:
        self._rules: dict[str, ValidationRules] = {}
        for rule_set in rules:
            self.register(rule_set)

    def register(self, rules: ValidationRules) -> None:
        existing = self._rules.get(rules.version)
        if existing is not None and existing != rules:
            raise ValueError(f"validation rules version {rules.version!r} is immutable")
        self._rules[rules.version] = rules

    def resolve(self, version: str | None) -> ValidationRules | None:
        return self._rules.get(version) if version else None


def _identifier_digest(values: Sequence[object]) -> str:
    canonical = json.dumps(sorted((str(value) for value in values)), separators=(",", ":"), ensure_ascii=False)
    return sha256(canonical.encode("utf-8")).hexdigest()


def _matches_type(value: object, value_type: FieldType) -> bool:
    if value_type is FieldType.STRING:
        return isinstance(value, str)
    if value_type is FieldType.INTEGER:
        return isinstance(value, int) and not isinstance(value, bool)
    if value_type is FieldType.NUMBER:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, bool)


def _domain_failure(rule: FieldRule, value: object) -> str | None:
    if not _matches_type(value, rule.value_type):
        return "type_mismatch"
    if rule.allowed_values is not None and value not in rule.allowed_values:
        return "value_not_allowed"
    if rule.minimum is not None and isinstance(value, (int, float)) and value < rule.minimum:
        return "value_below_minimum"
    if rule.maximum is not None and isinstance(value, (int, float)) and value > rule.maximum:
        return "value_above_maximum"
    return None


class DeterministicValidator:
    def __init__(self, registry: ValidationRulesRegistry, metadata_sink: ValidationMetadataSink) -> None:
        self._registry = registry
        self._metadata_sink = metadata_sink

    def validate(
        self,
        input_ids: Sequence[str],
        output: Sequence[Mapping[str, object]],
        context: ExecutionContext,
    ) -> ValidationResult:
        version = context.configuration.validation_rules_version
        rules = self._registry.resolve(version)
        if rules is None:
            return self._finish(
                version or "unavailable", input_ids, (), (),
                ("validation.rules.bound",), ("rules_unavailable",),
            )

        failed: list[str] = []
        reasons: list[str] = []
        output_ids: list[object] = []
        known_fields = {rule.field for rule in rules.field_rules}
        records: list[FrozenMapping] = []

        for index, record in enumerate(output):
            frozen = freeze(record)
            assert isinstance(frozen, FrozenMapping)
            records.append(frozen)
            identifier = record.get(rules.identifier_field)
            if identifier is None:
                self._add_failure(failed, reasons, "identifiers.present", "identifier_missing")
            else:
                output_ids.append(identifier)

            if not rules.allow_additional_fields:
                extra = set(record) - known_fields
                if extra:
                    self._add_failure(failed, reasons, "schema.additional_fields", "additional_fields")

            for rule in rules.field_rules:
                if rule.field not in record:
                    if rule.required:
                        self._add_failure(failed, reasons, rule.rule_id, "required_field_missing")
                    continue
                value = record[rule.field]
                if value is None:
                    if not rule.nullable:
                        self._add_failure(failed, reasons, rule.rule_id, "null_not_allowed")
                    continue
                reason = _domain_failure(rule, value)
                if reason:
                    self._add_failure(failed, reasons, rule.rule_id, reason)

        if len(output_ids) != len(set(map(str, output_ids))):
            self._add_failure(failed, reasons, "identifiers.unique", "duplicate_identifiers")
        if set(map(str, output_ids)) != set(input_ids):
            self._add_failure(failed, reasons, "identifiers.exact_set", "identifier_set_mismatch")
        if len(output) != len(input_ids):
            self._add_failure(failed, reasons, "cardinality.equal", "cardinality_mismatch")

        return self._finish(rules.version, input_ids, output_ids, records, tuple(failed), tuple(reasons))

    def _finish(
        self,
        version: str,
        input_ids: Sequence[str],
        output_ids: Sequence[object],
        records: Sequence[FrozenMapping],
        failed: tuple[str, ...],
        reasons: tuple[str, ...],
    ) -> ValidationResult:
        accepted = not failed
        status = ValidationStatus.ACCEPTED if accepted else ValidationStatus.VALIDATION_FAILED
        outcome = ValidationOutcome(
            accepted, status, failed, reasons,
            _identifier_digest(input_ids), _identifier_digest(output_ids), version,
        )
        metadata = ValidationMetadata(
            version, status, failed, reasons,
            outcome.input_identifier_digest, outcome.output_identifier_digest,
        )
        recorded = True
        try:
            self._metadata_sink.record(metadata)
        except Exception:
            recorded = False
        return ValidationResult(outcome, tuple(records) if accepted else (), recorded)

    @staticmethod
    def _add_failure(failed: list[str], reasons: list[str], rule_id: str, reason: str) -> None:
        if rule_id not in failed:
            failed.append(rule_id)
        if reason not in reasons:
            reasons.append(reason)
