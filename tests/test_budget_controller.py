from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest

from ai_retrieval.control_plane.budget import (
    BudgetBindingError,
    InMemoryBudgetController,
    ReservationStateError,
)
from ai_retrieval.domain.budget import (
    BudgetDimension,
    BudgetLimit,
    BudgetTerminalClassification,
    ReservationState,
    Usage,
)
from ai_retrieval.domain.configuration import ConfigurationReference, ExecutionConfiguration
from ai_retrieval.domain.execution import CancellationContext, DeadlineContext, ExecutionContext, ExecutionPath
from ai_retrieval.domain.identifiers import CorrelationId, ExecutionId
from ai_retrieval.domain.immutable import FrozenMapping


def context(execution_id: str = "execution-1", path: ExecutionPath = ExecutionPath.INTERACTIVE) -> ExecutionContext:
    configuration = ExecutionConfiguration(ConfigurationReference("default", "v1"), FrozenMapping(()))
    accepted_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return ExecutionContext(
        ExecutionId(execution_id),
        CorrelationId(f"correlation-{execution_id}"),
        path,
        configuration,
        DeadlineContext(accepted_at, None),
        CancellationContext(f"cancel-{execution_id}", 0),
    )


def controller() -> InMemoryBudgetController:
    identifiers = iter((f"reservation-{index}" for index in range(1, 20)))
    return InMemoryBudgetController(
        (
            BudgetLimit("request:1", 100, 1000),
            BudgetLimit("tenant:a", 500, 5000),
            BudgetLimit("provider:x", 1000, 10000),
        ),
        identifier=lambda: next(identifiers),
    )


def test_usage_requires_nonnegative_fixed_point_integer_units():
    with pytest.raises(ValueError):
        Usage(cost_minor_units=1.5)
    with pytest.raises(ValueError):
        Usage(input_tokens=-1)
    with pytest.raises(ValueError):
        Usage(output_tokens=True)


def test_binding_captures_all_scopes_and_is_immutable_for_execution():
    budgets = controller()
    bound = budgets.bind(context(), ("request:1", "tenant:a", "provider:x"))

    assert bound.configuration_version == "v1"
    assert bound.scopes == ("request:1", "tenant:a", "provider:x")
    assert budgets.bind(context(), bound.scopes) == bound
    with pytest.raises(BudgetBindingError):
        budgets.bind(context(), ("tenant:a",))


def test_reservation_is_all_or_nothing_and_names_every_exhausted_scope():
    budgets = controller()
    budgets.bind(context(), ("request:1", "tenant:a", "provider:x"))
    before = tuple(budgets.balance(scope) for scope in ("request:1", "tenant:a", "provider:x"))

    decision = budgets.reserve("execution-1", Usage(101, 4000, 2000))

    assert not decision.accepted
    assert decision.exhaustion.exhausted_scopes == ("request:1", "tenant:a")
    assert set(decision.exhaustion.dimensions) == {BudgetDimension.COST, BudgetDimension.TOKENS}
    assert tuple(budgets.balance(item.scope) for item in before) == before


def test_successful_reservation_atomically_debits_every_bound_scope():
    budgets = controller()
    scopes = ("request:1", "tenant:a", "provider:x")
    budgets.bind(context(), scopes)

    decision = budgets.reserve("execution-1", Usage(25, 100, 50))

    assert decision.accepted
    assert decision.reservation.scope_revisions == tuple((scope, 1) for scope in scopes)
    assert all(budgets.balance(scope).reserved == Usage(25, 100, 50) for scope in scopes)

def test_reconciliation_releases_unused_capacity_records_actual_once_and_drives_next_reservation():
    budgets = controller()
    scopes = ("request:1", "tenant:a")
    budgets.bind(context(), scopes)
    reservation = budgets.reserve("execution-1", Usage(80, 400, 200)).reservation

    receipt = budgets.reconcile(reservation.reservation_id, Usage(30, 100, 50))

    assert receipt.released == Usage(50, 300, 150)
    assert receipt.scope_revisions == (("request:1", 2), ("tenant:a", 2))
    assert budgets.reservation(reservation.reservation_id).state is ReservationState.RECONCILED
    for scope in scopes:
        balance = budgets.balance(scope)
        assert balance.reserved == Usage()
        assert balance.actual == Usage(30, 100, 50)
    assert budgets.reserve("execution-1", Usage(70, 700, 150)).accepted

    repeated = budgets.reconcile(reservation.reservation_id, Usage(30, 100, 50))
    assert repeated == receipt
    assert budgets.balance("request:1").actual == Usage(30, 100, 50)
    with pytest.raises(ReservationStateError):
        budgets.reconcile(reservation.reservation_id, Usage(31, 100, 50))


def test_reconciliation_records_actual_overrun_and_blocks_future_eligibility():
    budgets = controller()
    budgets.bind(context(), ("request:1", "tenant:a"))
    reservation = budgets.reserve("execution-1", Usage(50, 100, 100)).reservation

    budgets.reconcile(reservation.reservation_id, Usage(120, 600, 500))

    assert budgets.balance("request:1").available_cost_minor_units == -20
    assert not budgets.reserve("execution-1", Usage()).accepted
    assert budgets.balance("request:1").actual == Usage(120, 600, 500)


def test_interactive_and_bulk_exhaustion_projections_are_typed():
    budgets = controller()
    budgets.bind(context(), ("request:1",))
    exhaustion = budgets.reserve("execution-1", Usage(101, 0, 0)).exhaustion

    interactive = budgets.interactive_fallback(exhaustion)
    bulk = budgets.bulk_checkpoint_outcome(exhaustion)

    assert interactive.incomplete and interactive.exhausted_scopes == ("request:1",)
    assert bulk.checkpoint_unfinished_items
    assert bulk.terminal_classification is BudgetTerminalClassification.BUDGET_EXHAUSTED


def test_concurrent_reservations_cannot_overdraw_shared_scopes():
    budgets = InMemoryBudgetController((BudgetLimit("tenant:a", 100, 100),))
    for index in range(10):
        budgets.bind(context(f"execution-{index}"), ("tenant:a",))

    with ThreadPoolExecutor(max_workers=10) as pool:
        decisions = tuple(pool.map(lambda index: budgets.reserve(f"execution-{index}", Usage(30, 30, 0)), range(10)))

    assert sum(decision.accepted for decision in decisions) == 3
    balance = budgets.balance("tenant:a")
    assert balance.reserved == Usage(90, 90, 0)
    assert balance.available_cost_minor_units == 10
    assert balance.available_token_units == 10
