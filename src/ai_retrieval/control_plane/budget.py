"""Thread-safe in-memory atomic multi-scope budget controller."""

from dataclasses import dataclass
from threading import RLock
from typing import Callable, Iterable
from uuid import uuid4

from ai_retrieval.domain.budget import (
    BudgetBalance,
    BudgetBinding,
    BudgetDimension,
    BudgetExhaustion,
    BudgetLimit,
    BudgetReservation,
    BulkBudgetCheckpointOutcome,
    InteractiveBudgetFallback,
    ReconciliationReceipt,
    ReservationDecision,
    ReservationState,
    Usage,
)
from ai_retrieval.domain.execution import ExecutionContext


class BudgetError(RuntimeError):
    """Base class for typed budget-controller misuse."""


class BudgetBindingError(BudgetError):
    pass


class ReservationNotFoundError(BudgetError):
    pass


class ReservationStateError(BudgetError):
    pass


@dataclass
class _MutableBalance:
    limit: Usage
    reserved: Usage = Usage()
    actual: Usage = Usage()
    revision: int = 0


@dataclass
class _ReservationRecord:
    reservation: BudgetReservation
    receipt: ReconciliationReceipt | None = None


class InMemoryBudgetController:
    """Prototype ledger with one lock as its all-scope transaction boundary."""

    def __init__(self, limits: Iterable[BudgetLimit], identifier: Callable[[], str] | None = None) -> None:
        limits = tuple(limits)
        if len({limit.scope for limit in limits}) != len(limits):
            raise ValueError("budget scope limits must be unique")
        self._balances = {
            limit.scope: _MutableBalance(Usage(limit.cost_minor_units, limit.token_units, 0))
            for limit in limits
        }
        self._bindings: dict[str, BudgetBinding] = {}
        self._reservations: dict[str, _ReservationRecord] = {}
        self._identifier = identifier or (lambda: str(uuid4()))
        self._lock = RLock()

    def bind(self, context: ExecutionContext, scopes: Iterable[str]) -> BudgetBinding:
        execution_id = str(context.execution_id)
        normalized_scopes = self._normalize_scopes(scopes)
        with self._lock:
            self._require_known_scopes(normalized_scopes)
            proposed = BudgetBinding(execution_id, context.configuration.reference.version, normalized_scopes)
            existing = self._bindings.get(execution_id)
            if existing is not None and existing != proposed:
                raise BudgetBindingError("execution budget binding is immutable")
            self._bindings[execution_id] = proposed
            return proposed

    def binding_for(self, execution_id: str) -> BudgetBinding | None:
        with self._lock:
            return self._bindings.get(execution_id)

    def reserve(self, execution_id: str, estimate: Usage) -> ReservationDecision:
        """Atomically reserve estimate across every scope bound to the execution."""
        with self._lock:
            binding = self._bindings.get(execution_id)
            if binding is None:
                raise BudgetBindingError(f"execution has no budget binding: {execution_id}")
            exhaustion = self._exhaustion(binding.scopes, estimate)
            if exhaustion is not None:
                return ReservationDecision(exhaustion=exhaustion)

            revisions: list[tuple[str, int]] = []
            for scope in binding.scopes:
                balance = self._balances[scope]
                balance.reserved = _add(balance.reserved, estimate)
                balance.revision += 1
                revisions.append((scope, balance.revision))
            reservation = BudgetReservation(
                self._new_reservation_id(), execution_id, binding.scopes, estimate, tuple(revisions)
            )
            self._reservations[reservation.reservation_id] = _ReservationRecord(reservation)
            return ReservationDecision(reservation=reservation)

    def reconcile(self, reservation_id: str, actual: Usage) -> ReconciliationReceipt:
        """Atomically replace a reservation with actual usage in every scope."""
        with self._lock:
            record = self._reservations.get(reservation_id)
            if record is None:
                raise ReservationNotFoundError(f"unknown reservation: {reservation_id}")
            if record.receipt is not None:
                if record.receipt.actual != actual:
                    raise ReservationStateError("reservation already reconciled with different actual usage")
                return record.receipt

            reservation = record.reservation
            revisions: list[tuple[str, int]] = []
            for scope in reservation.scopes:
                balance = self._balances[scope]
                balance.reserved = _subtract(balance.reserved, reservation.estimate)
                balance.actual = _add(balance.actual, actual)
                balance.revision += 1
                revisions.append((scope, balance.revision))

            released = Usage(
                max(reservation.estimate.cost_minor_units - actual.cost_minor_units, 0),
                max(reservation.estimate.input_tokens - actual.input_tokens, 0),
                max(reservation.estimate.output_tokens - actual.output_tokens, 0),
            )
            receipt = ReconciliationReceipt(reservation_id, actual, released, tuple(revisions))
            record.reservation = BudgetReservation(
                reservation.reservation_id,
                reservation.execution_id,
                reservation.scopes,
                reservation.estimate,
                reservation.scope_revisions,
                ReservationState.RECONCILED,
            )
            record.receipt = receipt
            return receipt

    def balance(self, scope: str) -> BudgetBalance:
        with self._lock:
            self._require_known_scopes((scope,))
            balance = self._balances[scope]
            return BudgetBalance(scope, balance.limit, balance.reserved, balance.actual, balance.revision)

    def reservation(self, reservation_id: str) -> BudgetReservation:
        with self._lock:
            record = self._reservations.get(reservation_id)
            if record is None:
                raise ReservationNotFoundError(f"unknown reservation: {reservation_id}")
            return record.reservation

    @staticmethod
    def interactive_fallback(exhaustion: BudgetExhaustion) -> InteractiveBudgetFallback:
        return InteractiveBudgetFallback("budget_exhausted", exhaustion.exhausted_scopes)

    @staticmethod
    def bulk_checkpoint_outcome(exhaustion: BudgetExhaustion) -> BulkBudgetCheckpointOutcome:
        return BulkBudgetCheckpointOutcome("budget_exhausted", exhaustion.exhausted_scopes)

    def _exhaustion(self, scopes: tuple[str, ...], usage: Usage) -> BudgetExhaustion | None:
        exhausted_scopes: list[str] = []
        dimensions: set[BudgetDimension] = set()
        for scope in scopes:
            balance = self._balances[scope]
            cost_insufficient = usage.cost_minor_units > _available_cost(balance)
            tokens_insufficient = usage.total_tokens > _available_tokens(balance)
            if cost_insufficient or tokens_insufficient:
                exhausted_scopes.append(scope)
            if cost_insufficient:
                dimensions.add(BudgetDimension.COST)
            if tokens_insufficient:
                dimensions.add(BudgetDimension.TOKENS)
        if not exhausted_scopes:
            return None
        return BudgetExhaustion(tuple(exhausted_scopes), tuple(sorted(dimensions, key=str)))

    def _new_reservation_id(self) -> str:
        reservation_id = self._identifier()
        if not isinstance(reservation_id, str) or not reservation_id.strip():
            raise ValueError("reservation identifier must not be blank")
        if reservation_id in self._reservations:
            raise ValueError(f"duplicate reservation identifier: {reservation_id}")
        return reservation_id

    @staticmethod
    def _normalize_scopes(scopes: Iterable[str]) -> tuple[str, ...]:
        normalized = tuple(dict.fromkeys(scopes))
        if not normalized:
            raise BudgetBindingError("at least one applicable budget scope is required")
        return normalized

    def _require_known_scopes(self, scopes: tuple[str, ...]) -> None:
        missing = tuple(scope for scope in scopes if scope not in self._balances)
        if missing:
            raise BudgetBindingError(f"unknown budget scopes: {', '.join(missing)}")

def _add(left: Usage, right: Usage) -> Usage:
    return Usage(
        left.cost_minor_units + right.cost_minor_units,
        left.input_tokens + right.input_tokens,
        left.output_tokens + right.output_tokens,
    )


def _subtract(left: Usage, right: Usage) -> Usage:
    return Usage(
        left.cost_minor_units - right.cost_minor_units,
        left.input_tokens - right.input_tokens,
        left.output_tokens - right.output_tokens,
    )


def _available_cost(balance: _MutableBalance) -> int:
    return balance.limit.cost_minor_units - balance.reserved.cost_minor_units - balance.actual.cost_minor_units


def _available_tokens(balance: _MutableBalance) -> int:
    return balance.limit.total_tokens - balance.reserved.total_tokens - balance.actual.total_tokens
