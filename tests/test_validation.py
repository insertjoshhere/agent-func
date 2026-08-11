from datetime import datetime, timezone

from ai_retrieval.domain.configuration import ConfigurationReference, ExecutionConfiguration
from ai_retrieval.domain.execution import CancellationContext, DeadlineContext, ExecutionContext, ExecutionPath
from ai_retrieval.domain.identifiers import CorrelationId, ExecutionId
from ai_retrieval.domain.immutable import freeze
from ai_retrieval.validation import (
    DeterministicValidator, FieldRule, FieldType, ValidationRules,
    ValidationRulesRegistry, ValidationStatus,
)


class MetadataSink:
    def __init__(self, fail=False):
        self.fail = fail
        self.records = []

    def record(self, metadata):
        if self.fail:
            raise RuntimeError("telemetry unavailable")
        self.records.append(metadata)


def context(version="rules-1"):
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    config = ExecutionConfiguration(
        ConfigurationReference("default", "config-1"), freeze({}), "security-1", version
    )
    return ExecutionContext(
        ExecutionId("execution"), CorrelationId("correlation"), ExecutionPath.BULK,
        config, DeadlineContext(now, None), CancellationContext("cancel", 0),
    )


def rules(version="rules-1", allowed=frozenset({"ok", "review"})):
    return ValidationRules(version, "id", (
        FieldRule("schema.id", "id", FieldType.STRING),
        FieldRule("domain.label", "label", FieldType.STRING, allowed_values=allowed),
        FieldRule("domain.score", "score", FieldType.NUMBER, minimum=0, maximum=1),
    ))


def validator(sink=None, configured_rules=None):
    return DeterministicValidator(
        ValidationRulesRegistry(configured_rules or (rules(),)), sink or MetadataSink()
    )


def test_valid_output_is_accepted_with_bound_version_digests_and_metadata():
    sink = MetadataSink()
    result = validator(sink).validate(
        ("b", "a"),
        ({"id": "a", "label": "ok", "score": 0.5}, {"id": "b", "label": "review", "score": 1}),
        context(),
    )

    assert result.outcome.status is ValidationStatus.ACCEPTED
    assert tuple(record["id"] for record in result.accepted_records) == ("a", "b")
    assert result.outcome.input_identifier_digest == result.outcome.output_identifier_digest
    assert result.outcome.rules_version == "rules-1"
    assert result.metadata_recorded
    assert sink.records[0].rules_version == "rules-1"


def test_identical_input_output_and_bound_version_produce_identical_ordered_outcome_after_new_version_registered():
    registry = ValidationRulesRegistry((rules(),))
    sink = MetadataSink()
    checker = DeterministicValidator(registry, sink)
    output = ({"id": "a", "label": "bad", "score": 2},)

    first = checker.validate(("a",), output, context())
    registry.register(rules("rules-2", frozenset({"bad"})))
    second = checker.validate(("a",), output, context())

    assert first.outcome == second.outcome
    assert first.outcome.failed_rule_ids == ("domain.label", "domain.score")
    assert first.outcome.reason_codes == ("value_not_allowed", "value_above_maximum")


def test_missing_additional_duplicate_and_cardinality_mismatches_are_all_reported_and_output_withheld():
    result = validator().validate(
        ("a", "b"),
        ({"id": "a", "label": "ok", "score": 0.5, "extra": True},
         {"id": "a", "label": "ok"},
         {"label": "ok", "score": 0.5}),
        context(),
    )

    assert result.outcome.status is ValidationStatus.VALIDATION_FAILED
    assert not result.outcome.accepted
    assert result.accepted_records == ()
    assert set(result.outcome.failed_rule_ids) >= {
        "schema.additional_fields", "domain.score", "schema.id",
        "identifiers.present", "identifiers.unique", "identifiers.exact_set", "cardinality.equal",
    }


def test_missing_bound_rules_fail_closed_and_withhold_output():
    result = DeterministicValidator(ValidationRulesRegistry(), MetadataSink()).validate(
        ("a",), ({"id": "a"},), context("missing")
    )

    assert result.outcome.status is ValidationStatus.VALIDATION_FAILED
    assert result.outcome.failed_rule_ids == ("validation.rules.bound",)
    assert result.accepted_records == ()


def test_metadata_recording_failure_does_not_block_validated_processing():
    result = validator(MetadataSink(fail=True)).validate(
        ("a",), ({"id": "a", "label": "ok", "score": 0.5},), context()
    )

    assert result.outcome.accepted
    assert result.accepted_records[0]["id"] == "a"
    assert not result.metadata_recorded
