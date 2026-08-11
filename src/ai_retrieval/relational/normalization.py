"""Canonical relational value normalization and deterministic row ordering."""

from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from functools import cmp_to_key
from typing import Any

from ai_retrieval.relational.models import (
    NormalizedColumn,
    NormalizedResult,
    NormalizedType,
    NullPlacement,
    ProtectionMetadata,
    QueryPlan,
    RawRelationalResult,
    SortDirection,
)


class NormalizationError(ValueError):
    pass


def normalize_result(
    raw: RawRelationalResult,
    plan: QueryPlan,
    protection: ProtectionMetadata,
    adapter_version: str,
) -> NormalizedResult:
    columns = tuple(NormalizedColumn(item.name, item.value_type, item.nullable) for item in raw.columns)
    if len({item.name for item in columns}) != len(columns):
        raise NormalizationError("result column names must be unique")
    rows = tuple(_normalize_row(row, columns) for row in raw.rows)
    if plan.deterministic_order:
        indexes = _order_indexes(plan, columns)
        rows = tuple(sorted(rows, key=cmp_to_key(lambda left, right: _compare_rows(left, right, indexes))))
    return NormalizedResult(
        columns=columns,
        rows=rows,
        affected_rows=raw.affected_rows,
        transaction_outcome=raw.transaction_outcome,
        adapter_version=adapter_version,
        protection=protection,
    )


def _normalize_row(row: tuple[Any, ...], columns: tuple[NormalizedColumn, ...]) -> tuple[Any, ...]:
    if len(row) != len(columns):
        raise NormalizationError("row width does not match result columns")
    return tuple(_normalize_value(value, column) for value, column in zip(row, columns, strict=True))


def _normalize_value(value: Any, column: NormalizedColumn):
    if value is None:
        if not column.nullable:
            raise NormalizationError(f"column {column.name} is not nullable")
        return None
    expected = column.value_type
    if expected is NormalizedType.STRING and isinstance(value, str):
        return value
    if expected is NormalizedType.INTEGER and isinstance(value, int) and not isinstance(value, bool):
        return value
    if expected is NormalizedType.DECIMAL:
        try:
            decimal = value if isinstance(value, Decimal) else Decimal(str(value))
        except (InvalidOperation, ValueError):
            raise NormalizationError(f"column {column.name} is not decimal") from None
        if not decimal.is_finite():
            raise NormalizationError(f"column {column.name} must be finite")
        return decimal.normalize()
    if expected is NormalizedType.BOOLEAN and isinstance(value, bool):
        return value
    if expected is NormalizedType.DATE and isinstance(value, date) and not isinstance(value, datetime):
        return value
    if expected is NormalizedType.DATETIME and isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise NormalizationError(f"column {column.name} datetime must include timezone")
        return value.astimezone(timezone.utc)
    if expected is NormalizedType.BINARY and isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    raise NormalizationError(f"column {column.name} does not match {expected.value}")


def _order_indexes(plan: QueryPlan, columns: tuple[NormalizedColumn, ...]):
    by_name = {item.name: index for index, item in enumerate(columns)}
    try:
        return tuple((by_name[term.column], term) for term in plan.deterministic_order)
    except KeyError as error:
        raise NormalizationError(f"order column {error.args[0]} is absent from result") from None


def _compare_rows(left, right, indexes) -> int:
    for index, term in indexes:
        first, second = left[index], right[index]
        if first is None or second is None:
            comparison = 0 if first is second else (-1 if first is None else 1)
            if term.nulls is NullPlacement.LAST:
                comparison = -comparison
        else:
            comparison = (first > second) - (first < second)
            if term.direction is SortDirection.DESCENDING:
                comparison = -comparison
        if comparison:
            return comparison
    return 0
