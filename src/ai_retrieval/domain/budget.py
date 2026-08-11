"""Immutable fixed-point budget values and typed path outcomes."""

from dataclasses import dataclass
from enum import StrEnum


class BudgetDimension(StrEnum):
    COST = "cost"
    TOKENS = "tokens"


class ReservationState(StrEnum):
    ACTIVE = "active"
    RECONCILED = "reconciled"


class BudgetTerminalClassification(StrEnum):
    BUDGET_EXHAUSTED = "budget-exhausted"


def _require_nonnegative_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative fixed-point integer")


def _require_identifier(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be blank")


@dataclass(frozen=True)
class Usage:
    """Cost in minor currency units and separate input/output token counts."""

    cost_minor_units: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self) -> None:
        _require_nonnegative_integer("cost_minor_units", self.cost_minor_units)
        _require_nonnegative_integer("input_tokens", self.input_tokens)
        _require_nonnegative_integer("output_tokens", self.output_tokens)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

@dataclass(frozen=True)
class BudgetLimit:
    scope: str
    cost_minor_units: int
    token_units: int

    def __post_init__(self) -> None:
        _require_identifier("scope", self.scope)
        _require_nonnegative_integer("cost_minor_units", self.cost_minor_units)
        _require_nonnegative_integer("token_units", self.token_units)


@dataclass(frozen=True)
class BudgetBalance:
    scope: str
    limit: Usage
    reserved: Usage
    actual: Usage
    revision: int

    def __post_init__(self) -> None:
        _require_identifier("scope", self.scope)
        _require_nonnegative_integer("revision", self.revision)

    @property
    def available_cost_minor_units(self) -> int:
        return self.limit.cost_minor_units - self.reserved.cost_minor_units - self.actual.cost_minor_units

    @property
    def available_token_units(self) -> int:
        return self.limit.total_tokens - self.reserved.total_tokens - self.actual.total_tokens


@dataclass(frozen=True)
class BudgetBinding:
    execution_id: str
    configuration_version: str
    scopes: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_identifier("execution_id", self.execution_id)
        _require_identifier("configuration_version", self.configuration_version)
        if not self.scopes or len(set(self.scopes)) != len(self.scopes):
            raise ValueError("budget binding requires unique scopes")
        for scope in self.scopes:
            _require_identifier("scope", scope)


@dataclass(frozen=True)
class BudgetReservation:
    reservation_id: str
    execution_id: str
    scopes: tuple[str, ...]
    estimate: Usage
    scope_revisions: tuple[tuple[str, int], ...]
    state: ReservationState = ReservationState.ACTIVE

    def __post_init__(self) -> None:
        _require_identifier("reservation_id", self.reservation_id)
        _require_identifier("execution_id", self.execution_id)
        if not self.scopes:
            raise ValueError("reservation requires at least one scope")


@dataclass(frozen=True)
class BudgetExhaustion:
    exhausted_scopes: tuple[str, ...]
    dimensions: tuple[BudgetDimension, ...]

    def __post_init__(self) -> None:
        if not self.exhausted_scopes:
            raise ValueError("budget exhaustion requires an exhausted scope")


@dataclass(frozen=True)
class ReservationDecision:
    reservation: BudgetReservation | None = None
    exhaustion: BudgetExhaustion | None = None

    def __post_init__(self) -> None:
        if (self.reservation is None) == (self.exhaustion is None):
            raise ValueError("reservation decision requires exactly one outcome")

    @property
    def accepted(self) -> bool:
        return self.reservation is not None


@dataclass(frozen=True)
class ReconciliationReceipt:
    reservation_id: str
    actual: Usage
    released: Usage
    scope_revisions: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        _require_identifier("reservation_id", self.reservation_id)


@dataclass(frozen=True)
class InteractiveBudgetFallback:
    reason_code: str
    exhausted_scopes: tuple[str, ...]
    incomplete: bool = True


@dataclass(frozen=True)
class BulkBudgetCheckpointOutcome:
    reason_code: str
    exhausted_scopes: tuple[str, ...]
    checkpoint_unfinished_items: bool = True
    terminal_classification: BudgetTerminalClassification = BudgetTerminalClassification.BUDGET_EXHAUSTED
