"""Allowlisted plan registry, AST classification, and typed parameter binding."""

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal
from types import MappingProxyType

from ai_retrieval.relational.models import (
    MUTATION_OPERATIONS,
    RETRIEVAL_OPERATIONS,
    NormalizedType,
    OperationClassification,
    OperationNode,
    ParameterSpec,
    QueryPlan,
    QueryPlanReference,
)


class QueryPlanRegistry:
    """Stores immutable, explicitly versioned plans; registration is allowlisting."""

    def __init__(self, plans: Sequence[QueryPlan] = ()) -> None:
        self._plans: dict[QueryPlanReference, QueryPlan] = {}
        for plan in plans:
            self.register(plan)

    def register(self, plan: QueryPlan) -> None:
        reference = plan.reference
        existing = self._plans.get(reference)
        if existing is not None and existing != plan:
            raise ValueError(f"query plan {reference.plan_id}@{reference.version} is immutable")
        self._plans[reference] = plan

    def resolve(self, reference: QueryPlanReference) -> QueryPlan | None:
        return self._plans.get(reference)


def classify_operation(operation: OperationNode) -> OperationClassification:
    kinds = _operation_kinds(operation)
    if kinds & MUTATION_OPERATIONS:
        return OperationClassification.MUTATION
    if not kinds or not kinds <= RETRIEVAL_OPERATIONS:
        return OperationClassification.UNSUPPORTED
    return OperationClassification.RETRIEVAL


def operation_kinds(operation: OperationNode) -> frozenset:
    return frozenset(_operation_kinds(operation))


def bind_parameters(plan: QueryPlan, supplied: Mapping[str, object]) -> Mapping[str, object]:
    schema = {item.name: item for item in plan.parameter_schema}
    unknown = set(supplied) - set(schema)
    missing = {item.name for item in plan.parameter_schema if item.required and item.name not in supplied}
    if unknown:
        raise ValueError(f"unknown parameters: {', '.join(sorted(unknown))}")
    if missing:
        raise ValueError(f"missing parameters: {', '.join(sorted(missing))}")
    bound: dict[str, object] = {}
    for name, value in supplied.items():
        specification = schema[name]
        if value is None:
            if not specification.nullable:
                raise ValueError(f"parameter {name} must not be null")
        elif not _matches_type(value, specification.value_type):
            raise ValueError(f"parameter {name} must be {specification.value_type.value}")
        bound[name] = value
    return MappingProxyType(bound)


def _operation_kinds(operation: OperationNode) -> set:
    kinds = {operation.kind}
    for child in operation.children:
        kinds.update(_operation_kinds(child))
    return kinds


def _matches_type(value: object, expected: NormalizedType) -> bool:
    if expected is NormalizedType.STRING:
        return isinstance(value, str)
    if expected is NormalizedType.INTEGER:
        return isinstance(value, int) and not isinstance(value, bool)
    if expected is NormalizedType.DECIMAL:
        return isinstance(value, (int, Decimal)) and not isinstance(value, bool)
    if expected is NormalizedType.BOOLEAN:
        return isinstance(value, bool)
    if expected is NormalizedType.DATE:
        return isinstance(value, date) and not isinstance(value, datetime)
    if expected is NormalizedType.DATETIME:
        return isinstance(value, datetime)
    if expected is NormalizedType.BINARY:
        return isinstance(value, bytes)
    return False
