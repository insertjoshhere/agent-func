"""Configuration validation, immutable activation, binding, and artifact contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import math
from typing import Any, Protocol

from ai_retrieval.domain.configuration import (
    AdapterContractEvidence,
    ArtifactKind,
    BoundArtifact,
    ConfigurationReference,
    ConfigurationValidationFailure,
    ConfigurationValidationReport,
    CredentialAuthorizationEvidence,
    ExecutionConfiguration,
    ValidationFailureKind,
)
from ai_retrieval.domain.execution import ExecutionContext
from ai_retrieval.domain.failures import FailureCode, TypedFailure
from ai_retrieval.domain.immutable import FrozenMapping, FrozenValue, freeze


_REQUIRED_SECTIONS = (
    "routing",
    "database",
    "model",
    "budget",
    "deadline",
    "cancellation",
    "concurrency",
    "retry",
    "security",
    "validation",
    "telemetry",
    "write_back",
)
_MUTATION_OPERATIONS = frozenset(
    {"insert", "update", "delete", "merge", "alter", "create", "drop", "truncate", "grant", "revoke", "execute_mutating_routine"}
)


class ConfigurationBindingError(LookupError):
    def __init__(self, failure: TypedFailure) -> None:
        super().__init__(failure.message)
        self.failure = failure


class ConfigurationActivationError(ValueError):
    def __init__(self, report: ConfigurationValidationReport) -> None:
        super().__init__("configuration activation rejected")
        self.report = report


class EvidenceActivationError(ValueError):
    def __init__(self, report: ConfigurationValidationReport) -> None:
        super().__init__("adapter evidence rejected")
        self.report = report


class ConfigurationBinder(Protocol):
    def bind(self, reference: ConfigurationReference) -> ExecutionConfiguration: ...


class VersionRegistry:
    """A small activation registry for immutable policy/model/rule versions."""

    def __init__(self, versions: Sequence[str] = ()) -> None:
        self._versions = set(versions)

    def register(self, version: str) -> None:
        if not version.strip():
            raise ValueError("version must not be blank")
        self._versions.add(version)

    def contains(self, version: object) -> bool:
        return isinstance(version, str) and version in self._versions


class CredentialAuthorizationRegistry:
    def __init__(self, evidence: Sequence[CredentialAuthorizationEvidence] = ()) -> None:
        self._evidence = {item.credential_id: item for item in evidence}

    def register(self, evidence: CredentialAuthorizationEvidence) -> None:
        self._evidence[evidence.credential_id] = evidence

    def get(self, credential_id: object) -> CredentialAuthorizationEvidence | None:
        return self._evidence.get(credential_id) if isinstance(credential_id, str) else None


class AdapterEvidenceRegistry:
    """Activates adapters only from complete, fully passing contract evidence."""

    def __init__(self) -> None:
        self._supported: dict[tuple[str, str], AdapterContractEvidence] = {}

    def activate(self, evidence: AdapterContractEvidence) -> None:
        failures: list[ConfigurationValidationFailure] = []
        if not evidence.contract_suite_version.strip():
            failures.append(
                _failure(
                    "adapter_evidence.contract_suite_version.required",
                    "contract_suite_version",
                    ValidationFailureKind.MISSING,
                    "contract_suite_version is required",
                )
            )
        if evidence.evidence_timestamp.tzinfo is None or evidence.evidence_timestamp.utcoffset() is None:
            failures.append(
                _failure(
                    "adapter_evidence.timestamp.aware",
                    "evidence_timestamp",
                    ValidationFailureKind.INVALID,
                    "evidence_timestamp must include a timezone",
                )
            )

        required_cases: dict[str, Sequence[object]] = {
            "tested_operations": evidence.tested_operations,
            "normalized_type_cases": evidence.normalized_type_cases,
            "null_cases": evidence.null_cases,
            "ordering_cases": evidence.ordering_cases,
            "failure_cases": evidence.failure_cases,
            "commit_cases": evidence.commit_cases,
            "rollback_cases": evidence.rollback_cases,
            "test_outcomes": evidence.test_outcomes,
        }
        for field, cases in required_cases.items():
            if not cases:
                failures.append(_failure(f"adapter_evidence.{field}.required", field, ValidationFailureKind.MISSING, f"{field} is required"))

        blank_case_fields = (
            field
            for field, cases in required_cases.items()
            if field != "test_outcomes" and any(not isinstance(case, str) or not case.strip() for case in cases)
        )
        for field in blank_case_fields:
            failures.append(
                _failure(
                    f"adapter_evidence.{field}.valid",
                    field,
                    ValidationFailureKind.INVALID,
                    f"{field} must contain only nonblank case identifiers",
                )
            )

        outcome_names = tuple(name for name, _ in evidence.test_outcomes)
        if any(not name.strip() for name in outcome_names) or len(set(outcome_names)) != len(outcome_names):
            failures.append(
                _failure(
                    "adapter_evidence.test_outcomes.valid",
                    "test_outcomes",
                    ValidationFailureKind.INVALID,
                    "test outcomes require unique nonblank case identifiers",
                )
            )
        failed_tests = tuple(name for name, passed in evidence.test_outcomes if not passed)
        if failed_tests:
            failures.append(
                _failure(
                    "adapter_evidence.tests.passed",
                    "test_outcomes",
                    ValidationFailureKind.INVALID,
                    f"failed contract tests: {', '.join(sorted(failed_tests))}",
                )
            )
        if failures:
            raise EvidenceActivationError(ConfigurationValidationReport(tuple(sorted(set(failures)))))
        self._supported[(evidence.adapter_id, evidence.adapter_version)] = evidence

    def evidence_for(self, adapter_id: object, adapter_version: object) -> AdapterContractEvidence | None:
        if not isinstance(adapter_id, str) or not isinstance(adapter_version, str):
            return None
        return self._supported.get((adapter_id, adapter_version))


@dataclass(frozen=True)
class StoredConfigurationVersion:
    reference: ConfigurationReference
    content: FrozenMapping
    canonical_digest: str
    created_at: datetime
    retain_until: datetime


class ConfigurationValidator:
    """Evaluates every independent rule and returns all failures at once."""

    def __init__(
        self,
        security_policies: VersionRegistry,
        validation_rules: VersionRegistry,
        models: VersionRegistry,
        adapters: AdapterEvidenceRegistry,
        credentials: CredentialAuthorizationRegistry,
    ) -> None:
        self._security_policies = security_policies
        self._validation_rules = validation_rules
        self._models = models
        self._adapters = adapters
        self._credentials = credentials

    def validate(self, content: Mapping[str, Any]) -> ConfigurationValidationReport:
        failures: list[ConfigurationValidationFailure] = []
        sections: dict[str, Mapping[str, Any]] = {}
        for name in _REQUIRED_SECTIONS:
            value = content.get(name)
            if value is None:
                failures.append(_failure(f"profile.{name}.required", name, ValidationFailureKind.MISSING, f"{name} settings are required"))
            elif not isinstance(value, Mapping):
                failures.append(_failure(f"profile.{name}.mapping", name, ValidationFailureKind.INVALID, f"{name} must be a mapping"))
            else:
                sections[name] = value

        self._positive_number(sections, "deadline", "duration_seconds", failures)
        self._nonnegative_number(sections, "cancellation", "timeout_seconds", failures)
        self._positive_integer(sections, "concurrency", "limit", failures)
        self._positive_integer(sections, "retry", "max_attempts", failures)
        self._nonnegative_number(sections, "budget", "cost_limit", failures)
        self._nonnegative_number(sections, "budget", "token_limit", failures)
        self._boolean(sections, "routing", "interactive", failures)
        self._boolean(sections, "routing", "bulk", failures)
        self._boolean(sections, "telemetry", "enabled", failures)
        self._boolean(sections, "write_back", "enabled", failures)

        security_version = self._required_string(sections, "security", "policy_version", failures)
        if security_version is not None and not self._security_policies.contains(security_version):
            failures.append(_failure("profile.security.policy.available", "security.policy_version", ValidationFailureKind.UNAVAILABLE, "referenced security policy is unavailable"))
        validation_version = self._required_string(sections, "validation", "rules_version", failures)
        if validation_version is not None and not self._validation_rules.contains(validation_version):
            failures.append(_failure("profile.validation.rules.available", "validation.rules_version", ValidationFailureKind.UNAVAILABLE, "referenced validation rules are unavailable"))

        model_versions = sections.get("model", {}).get("versions")
        if not isinstance(model_versions, (list, tuple)) or not model_versions:
            failures.append(_failure("profile.model.versions.required", "model.versions", ValidationFailureKind.INVALID, "at least one model version is required"))
        else:
            for index, version in enumerate(model_versions):
                if not isinstance(version, str) or not version.strip():
                    failures.append(_failure("profile.model.version.valid", f"model.versions[{index}]", ValidationFailureKind.INVALID, "model version must be a nonblank string"))
                elif not self._models.contains(version):
                    failures.append(_failure("profile.model.version.available", f"model.versions[{index}]", ValidationFailureKind.UNAVAILABLE, f"model version {version} is unavailable"))

        database = sections.get("database", {})
        adapter_id = database.get("adapter_id")
        adapter_version = database.get("adapter_version")
        if not isinstance(adapter_id, str) or not adapter_id.strip():
            failures.append(_failure("profile.database.adapter_id.required", "database.adapter_id", ValidationFailureKind.INVALID, "adapter_id is required"))
        if not isinstance(adapter_version, str) or not adapter_version.strip():
            failures.append(_failure("profile.database.adapter_version.required", "database.adapter_version", ValidationFailureKind.INVALID, "adapter_version is required"))
        if isinstance(adapter_id, str) and adapter_id.strip() and isinstance(adapter_version, str) and adapter_version.strip():
            if self._adapters.evidence_for(adapter_id, adapter_version) is None:
                failures.append(_failure("profile.database.adapter.evidence", "database.adapter_version", ValidationFailureKind.UNAVAILABLE, "complete passing adapter evidence is unavailable"))

        credential_id = database.get("read_only_credential_id")
        if not isinstance(credential_id, str) or not credential_id.strip():
            failures.append(_failure("profile.database.credential.required", "database.read_only_credential_id", ValidationFailureKind.INVALID, "read-only credential is required"))
        else:
            authorization = self._credentials.get(credential_id)
            if authorization is None:
                failures.append(_failure("profile.database.credential.evidence", "database.read_only_credential_id", ValidationFailureKind.UNAVAILABLE, "credential authorization evidence is unavailable"))
            elif frozenset(operation.casefold() for operation in authorization.effective_operations) & _MUTATION_OPERATIONS:
                failures.append(_failure("profile.database.credential.read_only", "database.read_only_credential_id", ValidationFailureKind.UNAUTHORIZED, "credential has effective mutation authorization"))

        try:
            canonicalize_profile(content)
        except (TypeError, ValueError) as error:
            failures.append(_failure("profile.content.canonical", "$", ValidationFailureKind.INVALID, str(error)))
        return ConfigurationValidationReport(tuple(sorted(set(failures))))

    @staticmethod
    def _required_string(sections: Mapping[str, Mapping[str, Any]], section: str, field: str, failures: list[ConfigurationValidationFailure]) -> str | None:
        value = sections.get(section, {}).get(field)
        if not isinstance(value, str) or not value.strip():
            failures.append(_failure(f"profile.{section}.{field}.required", f"{section}.{field}", ValidationFailureKind.INVALID, f"{field} must be a nonblank string"))
            return None
        return value

    @staticmethod
    def _positive_number(sections: Mapping[str, Mapping[str, Any]], section: str, field: str, failures: list[ConfigurationValidationFailure]) -> None:
        _validate_number(sections, section, field, failures, lambda value: value > 0, "must be greater than zero")

    @staticmethod
    def _nonnegative_number(sections: Mapping[str, Mapping[str, Any]], section: str, field: str, failures: list[ConfigurationValidationFailure]) -> None:
        _validate_number(sections, section, field, failures, lambda value: value >= 0, "must be non-negative")

    @staticmethod
    def _positive_integer(sections: Mapping[str, Mapping[str, Any]], section: str, field: str, failures: list[ConfigurationValidationFailure]) -> None:
        value = sections.get(section, {}).get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            failures.append(_failure(f"profile.{section}.{field}.valid", f"{section}.{field}", ValidationFailureKind.INVALID, f"{field} must be a positive integer"))

    @staticmethod
    def _boolean(sections: Mapping[str, Mapping[str, Any]], section: str, field: str, failures: list[ConfigurationValidationFailure]) -> None:
        if section not in sections:
            return
        if not isinstance(sections[section].get(field), bool):
            failures.append(_failure(f"profile.{section}.{field}.valid", f"{section}.{field}", ValidationFailureKind.INVALID, f"{field} must be boolean"))


class ConfigurationRegistry(ConfigurationBinder):
    """Validates then stores content-addressed immutable profile versions."""

    def __init__(
        self,
        validator: ConfigurationValidator,
        retention_period: timedelta,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if retention_period.total_seconds() < 0:
            raise ValueError("retention period must be non-negative")
        self._validator = validator
        self._retention_period = retention_period
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._versions: dict[ConfigurationReference, StoredConfigurationVersion] = {}
        self._active: dict[str, ConfigurationReference] = {}

    def activate(self, profile_id: str, content: Mapping[str, Any]) -> ConfigurationReference:
        if not profile_id.strip():
            raise ValueError("profile_id must not be blank")
        report = self._validator.validate(content)
        if not report.valid:
            raise ConfigurationActivationError(report)
        canonical, digest = canonicalize_profile(content)
        reference = ConfigurationReference(profile_id, f"sha256:{digest}")
        existing = self._versions.get(reference)
        if existing is not None and existing.content != canonical:
            raise RuntimeError("configuration digest collision")
        if existing is None:
            created_at = self._clock()
            self._versions[reference] = StoredConfigurationVersion(reference, canonical, digest, created_at, created_at + self._retention_period)
        self._active[profile_id] = reference
        return reference

    def active_reference(self, profile_id: str) -> ConfigurationReference:
        try:
            return self._active[profile_id]
        except KeyError as error:
            raise ConfigurationBindingError(TypedFailure(FailureCode.CONFIGURATION_UNAVAILABLE, f"active configuration {profile_id} is unavailable")) from error

    def stored(self, reference: ConfigurationReference) -> StoredConfigurationVersion:
        try:
            return self._versions[reference]
        except KeyError as error:
            raise ConfigurationBindingError(TypedFailure(FailureCode.CONFIGURATION_UNAVAILABLE, f"configuration {reference.profile_id}@{reference.version} is unavailable")) from error

    def bind(self, reference: ConfigurationReference) -> ExecutionConfiguration:
        stored = self.stored(reference)
        return _execution_configuration(reference, stored.content)


class BoundArtifactFactory:
    """Creates result/checkpoint/effect/telemetry/audit records from bound context."""

    def create(self, context: ExecutionContext, artifact_id: str, kind: ArtifactKind, content: Mapping[str, Any]) -> BoundArtifact:
        frozen = freeze(content)
        if not isinstance(frozen, FrozenMapping):
            raise TypeError("artifact content must be a mapping")
        return BoundArtifact(artifact_id, kind, context.configuration.reference, frozen)


class StaticConfigurationBinder:
    """Copies known profile content so later source mutation cannot affect executions."""

    def __init__(self, profiles: Mapping[ConfigurationReference, Mapping[str, Any]]) -> None:
        self._profiles = profiles

    def bind(self, reference: ConfigurationReference) -> ExecutionConfiguration:
        try:
            source = self._profiles[reference]
        except KeyError as error:
            failure = TypedFailure(FailureCode.CONFIGURATION_UNAVAILABLE, f"configuration {reference.profile_id}@{reference.version} is unavailable")
            raise ConfigurationBindingError(failure) from error
        content = freeze(source)
        assert isinstance(content, FrozenMapping)
        return _execution_configuration(reference, content)


def canonicalize_profile(content: Mapping[str, Any]) -> tuple[FrozenMapping, str]:
    frozen = freeze(content)
    if not isinstance(frozen, FrozenMapping):
        raise TypeError("configuration profile must be a mapping")
    canonical_json = json.dumps(_canonical_value(frozen), ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)
    return frozen, sha256(canonical_json.encode("utf-8")).hexdigest()


def _canonical_value(value: FrozenValue) -> Any:
    if isinstance(value, FrozenMapping):
        return {key: _canonical_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_canonical_value(item) for item in value]
    if isinstance(value, frozenset):
        items = [_canonical_value(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("configuration numbers must be finite")
    return value


def _execution_configuration(reference: ConfigurationReference, content: FrozenMapping) -> ExecutionConfiguration:
    security = content.get("security")
    validation = content.get("validation")
    security_version = security.get("policy_version") if isinstance(security, FrozenMapping) else content.get("security_policy_version")
    validation_version = validation.get("rules_version") if isinstance(validation, FrozenMapping) else content.get("validation_rules_version")
    return ExecutionConfiguration(
        reference=reference,
        content=content,
        security_policy_version=security_version if isinstance(security_version, str) else None,
        validation_rules_version=validation_version if isinstance(validation_version, str) else None,
    )


def _validate_number(
    sections: Mapping[str, Mapping[str, Any]],
    section: str,
    field: str,
    failures: list[ConfigurationValidationFailure],
    predicate: Callable[[float], bool],
    message: str,
) -> None:
    value = sections.get(section, {}).get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not predicate(value):
        failures.append(_failure(f"profile.{section}.{field}.valid", f"{section}.{field}", ValidationFailureKind.INVALID, f"{field} {message}"))


def _failure(rule_id: str, path: str, kind: ValidationFailureKind, message: str) -> ConfigurationValidationFailure:
    return ConfigurationValidationFailure(rule_id, path, kind, message)
