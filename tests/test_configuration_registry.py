from datetime import datetime, timedelta, timezone

import pytest

from ai_retrieval.control_plane.configuration import (
    AdapterEvidenceRegistry,
    BoundArtifactFactory,
    ConfigurationActivationError,
    ConfigurationRegistry,
    ConfigurationValidator,
    CredentialAuthorizationRegistry,
    EvidenceActivationError,
    VersionRegistry,
)
from ai_retrieval.domain.configuration import (
    AdapterContractEvidence,
    ArtifactKind,
    CredentialAuthorizationEvidence,
)
from ai_retrieval.domain.execution import CancellationContext, DeadlineContext, ExecutionContext, ExecutionPath
from ai_retrieval.domain.identifiers import CorrelationId, ExecutionId


def adapter_evidence(*, passed: bool = True) -> AdapterContractEvidence:
    return AdapterContractEvidence(
        adapter_id="memory",
        adapter_version="1",
        contract_suite_version="contract-1",
        tested_operations=("read",),
        normalized_type_cases=("integer",),
        null_cases=("null",),
        ordering_cases=("explicit",),
        failure_cases=("unavailable",),
        commit_cases=("commit",),
        rollback_cases=("rollback",),
        test_outcomes=(("read", passed),),
        evidence_timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )


def valid_profile() -> dict:
    return {
        "routing": {"interactive": True, "bulk": True},
        "database": {"adapter_id": "memory", "adapter_version": "1", "read_only_credential_id": "reader"},
        "model": {"versions": ["model-1"]},
        "budget": {"cost_limit": 10, "token_limit": 1000},
        "deadline": {"duration_seconds": 2},
        "cancellation": {"timeout_seconds": 0},
        "concurrency": {"limit": 4},
        "retry": {"max_attempts": 2},
        "security": {"policy_version": "security-1"},
        "validation": {"rules_version": "rules-1"},
        "telemetry": {"enabled": True},
        "write_back": {"enabled": False},
    }


@pytest.fixture
def registry() -> ConfigurationRegistry:
    adapters = AdapterEvidenceRegistry()
    adapters.activate(adapter_evidence())
    validator = ConfigurationValidator(
        VersionRegistry(("security-1",)),
        VersionRegistry(("rules-1",)),
        VersionRegistry(("model-1",)),
        adapters,
        CredentialAuthorizationRegistry((CredentialAuthorizationEvidence("reader", frozenset({"select"})),)),
    )
    return ConfigurationRegistry(
        validator,
        timedelta(days=30),
        clock=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc),
    )


def test_validation_aggregates_all_profile_and_reference_failures(registry):
    profile = valid_profile()
    del profile["routing"]
    profile["deadline"]["duration_seconds"] = 0
    profile["security"]["policy_version"] = "missing-policy"
    profile["model"]["versions"] = ["missing-model"]
    profile["database"]["adapter_version"] = "missing-adapter"
    profile["database"]["read_only_credential_id"] = "missing-credential"

    with pytest.raises(ConfigurationActivationError) as raised:
        registry.activate("default", profile)

    rule_ids = {failure.rule_id for failure in raised.value.report.failures}
    assert rule_ids == {
        "profile.routing.required",
        "profile.deadline.duration_seconds.valid",
        "profile.security.policy.available",
        "profile.model.version.available",
        "profile.database.adapter.evidence",
        "profile.database.credential.evidence",
    }


def test_canonical_content_reuses_version_and_changed_content_creates_new_immutable_version(registry):
    original = valid_profile()
    reordered = {key: original[key] for key in reversed(original)}

    first = registry.activate("default", original)
    same = registry.activate("default", reordered)
    original["concurrency"]["limit"] = 99
    bound = registry.bind(first)
    changed = registry.activate("default", original)

    assert first == same
    assert changed != first
    assert bound.content["concurrency"]["limit"] == 4
    assert registry.stored(first).retain_until.isoformat() == "2025-01-31T00:00:00+00:00"


def test_adapter_activation_rejects_failed_evidence():
    registry = AdapterEvidenceRegistry()

    with pytest.raises(EvidenceActivationError) as failed:
        registry.activate(adapter_evidence(passed=False))

    assert {item.rule_id for item in failed.value.report.failures} == {"adapter_evidence.tests.passed"}
    assert registry.evidence_for("memory", "1") is None


def test_adapter_activation_rejects_incomplete_evidence():
    registry = AdapterEvidenceRegistry()
    evidence = adapter_evidence()
    incomplete = AdapterContractEvidence(
        adapter_id=evidence.adapter_id,
        adapter_version=evidence.adapter_version,
        contract_suite_version="",
        tested_operations=(),
        normalized_type_cases=evidence.normalized_type_cases,
        null_cases=evidence.null_cases,
        ordering_cases=evidence.ordering_cases,
        failure_cases=evidence.failure_cases,
        commit_cases=evidence.commit_cases,
        rollback_cases=evidence.rollback_cases,
        test_outcomes=evidence.test_outcomes,
        evidence_timestamp=datetime(2025, 1, 1),
    )

    with pytest.raises(EvidenceActivationError) as failed:
        registry.activate(incomplete)

    assert {item.rule_id for item in failed.value.report.failures} == {
        "adapter_evidence.contract_suite_version.required",
        "adapter_evidence.tested_operations.required",
        "adapter_evidence.timestamp.aware",
    }


def test_validation_checks_required_routing_and_telemetry_settings(registry):
    profile = valid_profile()
    del profile["routing"]["interactive"]
    profile["routing"]["bulk"] = "yes"
    del profile["telemetry"]["enabled"]

    with pytest.raises(ConfigurationActivationError) as raised:
        registry.activate("default", profile)

    assert {failure.rule_id for failure in raised.value.report.failures} == {
        "profile.routing.interactive.valid",
        "profile.routing.bulk.valid",
        "profile.telemetry.enabled.valid",
    }


def test_every_artifact_kind_carries_bound_configuration_version(registry):
    reference = registry.activate("default", valid_profile())
    configuration = registry.bind(reference)
    context = ExecutionContext(
        ExecutionId("execution"),
        CorrelationId("correlation"),
        ExecutionPath.BULK,
        configuration,
        DeadlineContext(datetime(2025, 1, 1, tzinfo=timezone.utc), None),
        CancellationContext("cancel", 0),
    )

    for kind in ArtifactKind:
        artifact = BoundArtifactFactory().create(context, f"artifact-{kind.value}", kind, {"ok": True})
        assert artifact.configuration == reference
        assert artifact.configuration_version == reference.version


def test_mutation_capable_read_credential_is_unauthorized():
    adapters = AdapterEvidenceRegistry()
    adapters.activate(adapter_evidence())
    validator = ConfigurationValidator(
        VersionRegistry(("security-1",)),
        VersionRegistry(("rules-1",)),
        VersionRegistry(("model-1",)),
        adapters,
        CredentialAuthorizationRegistry((CredentialAuthorizationEvidence("reader", frozenset({"select", "UPDATE"})),)),
    )

    report = validator.validate(valid_profile())

    assert [failure.rule_id for failure in report.failures] == ["profile.database.credential.read_only"]
