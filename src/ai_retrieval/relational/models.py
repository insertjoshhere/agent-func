"""Immutable vendor-neutral relational plans, results, and security values."""

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any


class OperationKind(StrEnum):
    SELECT = "select"
    PROJECT = "project"
    FILTER = "filter"
    JOIN = "join"
    AGGREGATE = "aggregate"
    SORT = "sort"
    LIMIT = "limit"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    MERGE = "merge"
    ALTER = "alter"
    CREATE = "create"
    DROP = "drop"
    TRUNCATE = "truncate"
    GRANT = "grant"
    REVOKE = "revoke"
    CALL = "call"
    UNKNOWN = "unknown"


RETRIEVAL_OPERATIONS = frozenset({
    OperationKind.SELECT, OperationKind.PROJECT, OperationKind.FILTER,
    OperationKind.JOIN, OperationKind.AGGREGATE, OperationKind.SORT,
    OperationKind.LIMIT,
})
MUTATION_OPERATIONS = frozenset(set(OperationKind) - RETRIEVAL_OPERATIONS - {OperationKind.UNKNOWN})


@dataclass(frozen=True)
class OperationNode:
    kind: OperationKind
    children: tuple["OperationNode", ...] = ()
    dataset: str | None = None


class OperationClassification(StrEnum):
    RETRIEVAL = "retrieval"
    MUTATION = "mutation"
    UNSUPPORTED = "unsupported"


class NormalizedType(StrEnum):
    STRING = "string"
    INTEGER = "integer"
    DECIMAL = "decimal"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    BINARY = "binary"


NormalizedScalar = str | int | Decimal | bool | date | datetime | bytes | None


class SortDirection(StrEnum):
    ASCENDING = "ascending"
    DESCENDING = "descending"


class NullPlacement(StrEnum):
    FIRST = "first"
    LAST = "last"


@dataclass(frozen=True)
class QueryPlanReference:
    plan_id: str
    version: str

    def __post_init__(self) -> None:
        if not self.plan_id.strip() or not self.version.strip():
            raise ValueError("query plan identifier and version must not be blank")


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    value_type: NormalizedType
    required: bool = True
    nullable: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("parameter name must not be blank")


@dataclass(frozen=True)
class OrderTerm:
    column: str
    direction: SortDirection = SortDirection.ASCENDING
    nulls: NullPlacement = NullPlacement.LAST

    def __post_init__(self) -> None:
        if not self.column.strip():
            raise ValueError("order column must not be blank")


@dataclass(frozen=True)
class QueryPlan:
    reference: QueryPlanReference
    operation: OperationNode
    parameter_schema: tuple[ParameterSpec, ...]
    required_capabilities: frozenset[str]
    deterministic_order: tuple[OrderTerm, ...]
    intrinsically_ordered: bool
    max_rows: int
    max_bytes: int
    authorized_datasets: frozenset[str]

    def __post_init__(self) -> None:
        names = tuple(item.name for item in self.parameter_schema)
        if len(set(names)) != len(names):
            raise ValueError("query parameter names must be unique")
        if self.max_rows <= 0 or self.max_bytes <= 0:
            raise ValueError("query plan limits must be positive")
        if any(not value.strip() for value in self.required_capabilities | self.authorized_datasets):
            raise ValueError("capabilities and datasets must not be blank")


@dataclass(frozen=True)
class VendorNeutralContract:
    version: str
    capabilities: frozenset[str]
    normalized_types: frozenset[NormalizedType]

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("contract version must not be blank")


@dataclass(frozen=True)
class DatabaseAccessDecision:
    allowed: bool
    credential_id: str
    service_identity: str
    policy_version: str
    reason_code: str
    transport_encryption: str | None
    persistence_encryption: str | None


@dataclass(frozen=True)
class ProtectionMetadata:
    policy_version: str
    service_identity: str
    transport_encryption: str
    persistence_encryption: str


@dataclass(frozen=True)
class RawColumn:
    name: str
    value_type: NormalizedType
    nullable: bool = True


@dataclass(frozen=True)
class RawRelationalResult:
    columns: tuple[RawColumn, ...]
    rows: tuple[tuple[Any, ...], ...]
    affected_rows: int | None = None
    transaction_outcome: str | None = None


@dataclass(frozen=True)
class NormalizedColumn:
    name: str
    value_type: NormalizedType
    nullable: bool


@dataclass(frozen=True)
class NormalizedResult:
    columns: tuple[NormalizedColumn, ...]
    rows: tuple[tuple[NormalizedScalar, ...], ...]
    affected_rows: int | None
    transaction_outcome: str | None
    adapter_version: str
    protection: ProtectionMetadata


@dataclass(frozen=True)
class SecurityAuditEvent:
    correlation_id: str
    configuration_version: str
    policy_version: str
    service_identity: str
    decision: str
    reason_code: str
    timestamp: datetime
    operation_classification: OperationClassification
    plan_reference: QueryPlanReference
    redacted_details: str = "[REDACTED]"


@dataclass(frozen=True)
class ApprovedEffect:
    effect_id: str
    operation: OperationNode
    parameters: tuple[tuple[str, object], ...]


@dataclass(frozen=True)
class EffectOutcome:
    affected_rows: int
    transaction_outcome: str
    adapter_version: str
