"""Composed deterministic and invocation-specific task output validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
import json

from ai_retrieval.domain.execution import ExecutionContext
from ai_retrieval.domain.immutable import FrozenMapping
from ai_retrieval.tasks.models import PackedRequest, PreparedTask, TaskFunction
from ai_retrieval.validation import DeterministicValidator
from ai_retrieval.validation.models import (
    ValidationOutcome,
    ValidationResult,
    ValidationStatus,
)


_CLASSIFY_LABEL_RULE = "task_output.classify.label.membership"
_SUMMARY_WORD_RULE = "task_output.summarize.summary.word_limit"
_SUMMARY_CHARACTER_RULE = "task_output.summarize.summary.character_limit"


class TaskOutputValidator:
    """Compose bound deterministic validation with task-specific constraints."""

    def __init__(self, deterministic_validator: DeterministicValidator) -> None:
        self._deterministic_validator = deterministic_validator

    def validate(
        self,
        prepared: PreparedTask,
        pack: PackedRequest,
        output: Sequence[Mapping[str, object]],
        context: ExecutionContext,
    ) -> ValidationResult:
        """Validate one pack and return accepted records in pack input order."""
        self._require_pack(prepared, pack)
        base = self._deterministic_validator.validate(pack.input_ids, output, context)
        if not base.outcome.accepted:
            return base

        failed_rules, reasons = self._task_failures(prepared, base.accepted_records)
        if failed_rules:
            return _failed_result(base, failed_rules, reasons)
        return ValidationResult(
            base.outcome,
            _order_records(base.accepted_records, pack.input_ids),
            base.metadata_recorded,
        )

    validate_pack = validate

    def validate_packs(
        self,
        prepared: PreparedTask,
        outputs: Sequence[tuple[PackedRequest, Sequence[Mapping[str, object]]]],
        context: ExecutionContext,
    ) -> ValidationResult:
        """Validate all packs and concatenate only a complete successful result."""
        ordered = tuple(sorted(outputs, key=lambda item: item[0].pack_index))
        if tuple(pack.pack_index for pack, _ in ordered) != tuple(range(len(prepared.packs))):
            raise ValueError("pack outputs must contain every prepared pack exactly once")

        accepted: list[FrozenMapping] = []
        validated: list[ValidationResult] = []
        for expected, (pack, output) in zip(prepared.packs, ordered, strict=True):
            if pack != expected:
                raise ValueError("pack outputs must match the prepared task")
            result = self.validate(prepared, pack, output, context)
            validated.append(result)
            if result.outcome.accepted:
                accepted.extend(result.accepted_records)

        failed = tuple(result for result in validated if not result.outcome.accepted)
        if failed:
            return _combined_failed_result(prepared, validated)
        return ValidationResult(
            _combined_outcome(prepared, validated, accepted=True),
            tuple(accepted),
            all(result.metadata_recorded for result in validated),
        )

    @staticmethod
    def _require_pack(prepared: PreparedTask, pack: PackedRequest) -> None:
        if pack not in prepared.packs:
            raise ValueError("packed request does not belong to the prepared task")

    @staticmethod
    def _task_failures(
        prepared: PreparedTask,
        records: Sequence[FrozenMapping],
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        failed: list[str] = []
        reasons: list[str] = []
        if prepared.invocation.function is TaskFunction.CLASSIFY:
            labels = prepared.invocation.parameters.get("labels")
            allowed = frozenset(labels) if isinstance(labels, tuple) else frozenset()
            if any(record["label"] not in allowed for record in records):
                failed.append(_CLASSIFY_LABEL_RULE)
                reasons.append("label_not_allowed")
            return tuple(failed), tuple(reasons)

        max_words = prepared.invocation.parameters.get("max_words")
        limits = prepared.definition.summary_limits
        if not isinstance(max_words, int) or isinstance(max_words, bool) or limits is None:
            raise ValueError("prepared summary task has invalid invocation constraints")
        if any(len(record["summary"].split()) > max_words for record in records):
            failed.append(_SUMMARY_WORD_RULE)
            reasons.append("summary_word_limit_exceeded")
        if any(len(record["summary"]) > limits.max_summary_characters for record in records):
            failed.append(_SUMMARY_CHARACTER_RULE)
            reasons.append("summary_character_limit_exceeded")
        return tuple(failed), tuple(reasons)


def _order_records(
    records: Sequence[FrozenMapping], input_ids: Sequence[str]
) -> tuple[FrozenMapping, ...]:
    by_identifier = {record["id"]: record for record in records}
    return tuple(by_identifier[identifier] for identifier in input_ids)


def _failed_result(
    base: ValidationResult,
    failed_rules: tuple[str, ...],
    reasons: tuple[str, ...],
) -> ValidationResult:
    outcome = base.outcome
    return ValidationResult(
        ValidationOutcome(
            False,
            ValidationStatus.VALIDATION_FAILED,
            tuple(dict.fromkeys((*outcome.failed_rule_ids, *failed_rules))),
            tuple(dict.fromkeys((*outcome.reason_codes, *reasons))),
            outcome.input_identifier_digest,
            outcome.output_identifier_digest,
            outcome.rules_version,
        ),
        (),
        False,
    )


def _combined_failed_result(
    prepared: PreparedTask, results: Sequence[ValidationResult]
) -> ValidationResult:
    return ValidationResult(
        _combined_outcome(prepared, results, accepted=False),
        (),
        False,
    )


def _combined_outcome(
    prepared: PreparedTask,
    results: Sequence[ValidationResult],
    accepted: bool,
) -> ValidationOutcome:
    first = results[0].outcome
    failed_rules = tuple(dict.fromkeys(
        rule for result in results for rule in result.outcome.failed_rule_ids
    ))
    reasons = tuple(dict.fromkeys(
        reason for result in results for reason in result.outcome.reason_codes
    ))
    input_ids = tuple(
        identifier for pack in prepared.packs for identifier in pack.input_ids
    )
    input_digest = _identifier_digest(input_ids)
    output_digest = input_digest if accepted else _combined_output_digest(results)
    return ValidationOutcome(
        accepted,
        ValidationStatus.ACCEPTED if accepted else ValidationStatus.VALIDATION_FAILED,
        failed_rules,
        reasons,
        input_digest,
        output_digest,
        first.rules_version,
    )


def _combined_output_digest(results: Sequence[ValidationResult]) -> str:
    canonical = json.dumps(
        [result.outcome.output_identifier_digest for result in results],
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def _identifier_digest(values: Sequence[object]) -> str:
    canonical = json.dumps(
        sorted(str(value) for value in values),
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


__all__ = ["TaskOutputValidator"]
