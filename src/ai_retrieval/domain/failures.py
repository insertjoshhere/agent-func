"""Typed, machine-readable failures returned across contracts."""

from dataclasses import dataclass
from enum import StrEnum


class FailureCode(StrEnum):
    AMBIGUOUS_PATH = "ambiguous_path"
    MISSING_PATH = "missing_path"
    INVALID_DISCRIMINATOR = "invalid_discriminator"
    CONFIGURATION_UNAVAILABLE = "configuration_unavailable"
    QUERY_PLAN_UNAVAILABLE = "query_plan_unavailable"
    UNSUPPORTED_OPERATION = "unsupported_operation"
    MUTATION_BLOCKED = "mutation_blocked"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    NONDETERMINISTIC_ORDER = "nondeterministic_order"
    INVALID_QUERY_PARAMETERS = "invalid_query_parameters"
    READ_ONLY_CREDENTIAL_REQUIRED = "read_only_credential_required"
    DATABASE_AUTHENTICATION_FAILED = "database_authentication_failed"
    DATABASE_ACCESS_DENIED = "database_access_denied"
    DATABASE_PROTECTION_REQUIRED = "database_protection_required"
    RESULT_NORMALIZATION_FAILED = "result_normalization_failed"
    SECURITY_POLICY_UNAVAILABLE = "security_policy_unavailable"
    SECURITY_POLICY_INVALID = "security_policy_invalid"
    SECURITY_POLICY_DENIED = "security_policy_denied"
    SECURITY_PROVIDER_DENIED = "security_provider_denied"
    SECURITY_DATA_DENIED = "security_data_denied"
    SECURITY_PROTECTION_REQUIRED = "security_protection_required"
    MODEL_AUTHENTICATION_FAILED = "model_authentication_failed"


@dataclass(frozen=True)
class TypedFailure:
    code: FailureCode
    message: str
    retryable: bool = False
