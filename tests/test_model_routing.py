from datetime import datetime, timedelta, timezone

import pytest

from ai_retrieval.domain.execution import ExecutionPath
from ai_retrieval.domain.model_routing import (
    EligibilityReason,
    ModelCandidate,
    ModelOperation,
    PolicyDecision,
    RoutingPolicy,
)
from ai_retrieval.domain.work import ModelWork
from ai_retrieval.model_routing import (
    AdmissionFailure,
    AttemptState,
    CapacityLimit,
    CapacityScope,
    CapacityScopeKind,
    HierarchicalCapacity,
    ModelRouter,
    RetryController,
    RetryGateSnapshot,
    RetryPolicy,
    RetryTerminalReason,
    RetryTransitionKind,
    SchedulingOrder,
)


NOW = datetime(2025, 1, 1, tzinfo=timezone.utc)


class AvailableBudget:
    def __init__(self, denied_models=()):
        self.denied_models = frozenset(denied_models)

    def is_available(self, operation, candidate):
        return candidate.model_id not in self.denied_models


def candidate(
    model_id,
    *,
    provider="provider-a",
    cost=10,
    priority=1,
    latency=100,
    quality=0.9,
    capabilities=frozenset({"summarize"}),
):
    return ModelCandidate(
        model_id,
        provider,
        capabilities,
        frozenset({"internal"}),
        cost,
        priority,
        latency,
        quality,
    )


def operation(
    operation_id="op-1",
    *,
    tenant="tenant-a",
    scope="request-a",
    path=ExecutionPath.INTERACTIVE,
    deadline_ms=1000,
    tokens=50,
):
    return ModelOperation(
        operation_id,
        tenant,
        scope,
        path,
        ModelWork("summary", "payload://one", ("item-1",), frozenset({"summarize"})),
        tokens,
        minimum_quality=0.8,
        data_classes=frozenset({"internal"}),
        deadline=NOW + timedelta(milliseconds=deadline_ms),
        completion_reserve_ms=50,
    )


def policy(**overrides):
    values = {
        "version": "policy-1",
        "decision": PolicyDecision.ALLOW,
        "available": True,
        "valid": True,
        "allowed_provider_ids": frozenset({"provider-a", "provider-b"}),
        "allowed_data_classes": frozenset({"internal"}),
        "minimum_quality": 0.7,
    }
    values.update(overrides)
    return RoutingPolicy(**values)


def capacity(limit=10):
    return HierarchicalCapacity(
        {CapacityScope(CapacityScopeKind.SYSTEM, "system"): CapacityLimit(concurrency=limit)}
    )


def test_selection_filters_all_gates_then_uses_cost_priority_and_stable_id():
    router = ModelRouter(AvailableBudget({"budget-denied"}), capacity())
    candidates = (
        candidate("expensive", cost=20, priority=99),
        candidate("z-model", cost=10, priority=3),
        candidate("a-model", cost=10, priority=3),
        candidate("low-priority", cost=10, priority=1),
        candidate("budget-denied", cost=1),
        candidate("wrong-provider", provider="provider-x", cost=1),
        candidate("too-slow", cost=1, latency=1000),
    )

    decision = router.admit(operation(), candidates, policy(), NOW)

    assert decision.admitted
    assert decision.candidate.model_id == "a-model"
    reasons = {item.candidate.model_id: item.reasons for item in decision.eligibility}
    assert reasons["budget-denied"] == (EligibilityReason.BUDGET_INELIGIBLE,)
    assert reasons["wrong-provider"] == (EligibilityReason.PROVIDER_DENIED,)
    assert reasons["too-slow"] == (EligibilityReason.DEADLINE_INELIGIBLE,)


def test_missing_invalid_or_denied_security_policy_fails_closed():
    router = ModelRouter(AvailableBudget(), capacity())

    missing = router.admit(operation(), (candidate("model"),), None, NOW)
    invalid = router.admit(operation("op-2"), (candidate("model"),), policy(valid=False), NOW)
    denied = router.admit(
        operation("op-3"),
        (candidate("model"),),
        policy(decision=PolicyDecision.INDETERMINATE),
        NOW,
    )

    assert missing.failure is invalid.failure is denied.failure is AdmissionFailure.NO_ELIGIBLE_MODEL
    assert missing.eligibility[0].reasons == (EligibilityReason.POLICY_UNAVAILABLE,)
    assert invalid.eligibility[0].reasons == (EligibilityReason.POLICY_INVALID,)
    assert denied.eligibility[0].reasons == (EligibilityReason.POLICY_DENIED,)


def test_deadline_only_rejection_returns_typed_deadline_outcome_without_capacity_use():
    limits = capacity(limit=1)
    router = ModelRouter(AvailableBudget(), limits)

    decision = router.admit(operation(deadline_ms=99), (candidate("model", latency=50),), policy(), NOW)

    assert not decision.admitted
    assert decision.failure is AdmissionFailure.DEADLINE_INELIGIBLE
    assert limits.active_count(CapacityScope(CapacityScopeKind.SYSTEM, "system")) == 0


def test_hierarchical_capacity_is_atomic_and_never_exceeds_any_scope():
    system = CapacityScope(CapacityScopeKind.SYSTEM, "system")
    tenant = CapacityScope(CapacityScopeKind.TENANT, "tenant-a")
    provider = CapacityScope(CapacityScopeKind.PROVIDER, "provider-a")
    limits = HierarchicalCapacity(
        {
            system: CapacityLimit(concurrency=2),
            tenant: CapacityLimit(concurrency=1),
            provider: CapacityLimit(concurrency=2),
        }
    )
    router = ModelRouter(AvailableBudget(), limits)

    first = router.admit(operation("op-1"), (candidate("model"),), policy(), NOW)
    blocked = router.admit(operation("op-2", scope="request-b"), (candidate("model"),), policy(), NOW)

    assert first.admitted and blocked.failure is AdmissionFailure.CAPACITY_UNAVAILABLE
    assert limits.active_count(system) == 1
    assert limits.active_count(tenant) == 1
    assert router.complete(first.lease)
    assert limits.active_count(system) == limits.active_count(tenant) == 0


def test_rpm_and_tpm_capacity_remain_consumed_after_concurrency_release_until_window_expires():
    provider = CapacityScope(CapacityScopeKind.PROVIDER, "provider-a")
    limits = HierarchicalCapacity(
        {provider: CapacityLimit(concurrency=1, requests_per_minute=1, tokens_per_minute=50)}
    )
    router = ModelRouter(AvailableBudget(), limits)
    first = router.admit(operation("op-1", tokens=50, deadline_ms=120000), (candidate("model"),), policy(), NOW)
    assert first.admitted and router.complete(first.lease)

    within_window = router.admit(
        operation("op-2", scope="request-b", tokens=1, deadline_ms=120000),
        (candidate("model"),),
        policy(),
        NOW + timedelta(seconds=59),
    )
    after_window = router.admit(
        operation("op-3", scope="request-c", tokens=50, deadline_ms=120000),
        (candidate("model"),),
        policy(),
        NOW + timedelta(seconds=60),
    )

    assert within_window.failure is AdmissionFailure.CAPACITY_UNAVAILABLE
    assert after_window.admitted


def test_capacity_skips_ranked_candidate_that_cannot_fit_candidate_specific_scope():
    blocked_model = CapacityScope(CapacityScopeKind.MODEL, "cheap")
    open_model = CapacityScope(CapacityScopeKind.MODEL, "fallback")
    limits = HierarchicalCapacity(
        {blocked_model: CapacityLimit(concurrency=0), open_model: CapacityLimit(concurrency=1)}
    )
    router = ModelRouter(AvailableBudget(), limits)

    decision = router.admit(
        operation(),
        (candidate("cheap", cost=1), candidate("fallback", cost=2)),
        policy(),
        NOW,
    )

    assert decision.admitted and decision.candidate.model_id == "fallback"


def test_interactive_first_scheduler_admits_first_fitting_interactive_operation():
    router = ModelRouter(AvailableBudget(), capacity(), SchedulingOrder.INTERACTIVE_FIRST)
    router.enqueue(
        operation("bulk", scope="job", path=ExecutionPath.BULK),
        (candidate("model"),),
        policy(),
    )
    router.enqueue(operation("interactive"), (candidate("model"),), policy())

    decision = router.schedule_next(NOW)

    assert decision.admitted and decision.lease.operation_id == "interactive"


def test_tenant_round_robin_rotates_tenants_while_preserving_each_tenant_fifo():
    router = ModelRouter(AvailableBudget(), capacity(), SchedulingOrder.TENANT_ROUND_ROBIN)
    for operation_id, tenant in (("a1", "a"), ("a2", "a"), ("b1", "b"), ("b2", "b")):
        router.enqueue(operation(operation_id, tenant=tenant, scope=operation_id), (candidate("model"),), policy())

    admitted = []
    for _ in range(4):
        decision = router.schedule_next(NOW)
        admitted.append(decision.lease.operation_id)
        assert router.complete(decision.lease)

    assert admitted == ["a1", "b1", "a2", "b2"]


def test_retry_and_configured_fallback_transition_only_when_every_gate_passes():
    controller = RetryController()
    op = operation(deadline_ms=1000)
    candidates = {"fallback": candidate("fallback", latency=200)}
    transition = controller.next_transition(
        op,
        AttemptState("primary", 1),
        "rate_limit",
        RetryPolicy(frozenset({"rate_limit"}), 2, (100,), ("fallback",)),
        candidates,
        RetryGateSnapshot(True, True, True),
        NOW,
    )

    assert transition.kind is RetryTransitionKind.FALLBACK
    assert transition.model_id == "fallback" and transition.delay_ms == 100
    assert transition.invokes_model


@pytest.mark.parametrize(
    ("failure_class", "state", "gates", "deadline_ms", "expected"),
    [
        ("bad_request", AttemptState("primary", 1), RetryGateSnapshot(True, True, True), 1000, RetryTerminalReason.NONRETRYABLE_FAILURE),
        ("timeout", AttemptState("primary", 2), RetryGateSnapshot(True, True, True), 1000, RetryTerminalReason.ATTEMPTS_EXHAUSTED),
        ("timeout", AttemptState("primary", 1), RetryGateSnapshot(False, True, True), 1000, RetryTerminalReason.BUDGET_INELIGIBLE),
        ("timeout", AttemptState("primary", 1), RetryGateSnapshot(True, False, True), 1000, RetryTerminalReason.CAPACITY_UNAVAILABLE),
        ("timeout", AttemptState("primary", 1), RetryGateSnapshot(True, True, False), 1000, RetryTerminalReason.MODEL_INELIGIBLE),
        ("timeout", AttemptState("primary", 1), RetryGateSnapshot(True, True, True), 349, RetryTerminalReason.DEADLINE_INELIGIBLE),
    ],
)
def test_failed_retry_gate_is_terminal_and_never_invokes(
    failure_class, state, gates, deadline_ms, expected
):
    transition = RetryController().next_transition(
        operation(deadline_ms=deadline_ms),
        state,
        failure_class,
        RetryPolicy(frozenset({"timeout"}), 2, (100,), ("fallback",)),
        {"fallback": candidate("fallback", latency=200)},
        gates,
        NOW,
    )

    assert transition.kind is RetryTransitionKind.TERMINAL
    assert transition.terminal_reason is expected
    assert not transition.invokes_model
