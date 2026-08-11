"""Deterministic economical selection and configurable fair scheduling."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Protocol, Sequence

from ai_retrieval.domain.execution import ExecutionPath
from ai_retrieval.domain.model_routing import (
    CandidateEligibility,
    EligibilityReason,
    ModelCandidate,
    ModelOperation,
    PolicyDecision,
    RoutingPolicy,
)
from ai_retrieval.model_routing.capacity import CapacityLease, HierarchicalCapacity


class BudgetAvailability(Protocol):
    """Atomic budget implementations report whether an estimate remains reservable."""

    def is_available(self, operation: ModelOperation, candidate: ModelCandidate) -> bool: ...


class AdmissionFailure(StrEnum):
    NO_ELIGIBLE_MODEL = "no_eligible_model"
    DEADLINE_INELIGIBLE = "deadline_ineligible"
    CAPACITY_UNAVAILABLE = "capacity_unavailable"


@dataclass(frozen=True)
class ModelAdmission:
    candidate: ModelCandidate | None
    lease: CapacityLease | None
    failure: AdmissionFailure | None
    eligibility: tuple[CandidateEligibility, ...]

    @property
    def admitted(self) -> bool:
        return self.candidate is not None and self.lease is not None and self.failure is None


class SchedulingOrder(StrEnum):
    FIFO = "fifo"
    INTERACTIVE_FIRST = "interactive_first"
    TENANT_ROUND_ROBIN = "tenant_round_robin"


@dataclass(frozen=True)
class _QueuedOperation:
    sequence: int
    operation: ModelOperation
    candidates: tuple[ModelCandidate, ...]
    policy: RoutingPolicy | None


class ModelRouter:
    def __init__(
        self,
        budget_availability: BudgetAvailability,
        capacity: HierarchicalCapacity,
        scheduling_order: SchedulingOrder = SchedulingOrder.FIFO,
    ) -> None:
        self._budget = budget_availability
        self._capacity = capacity
        self._scheduling_order = scheduling_order
        self._queue: list[_QueuedOperation] = []
        self._next_sequence = 0
        self._last_tenant: str | None = None

    def evaluate(
        self,
        operation: ModelOperation,
        candidates: Sequence[ModelCandidate],
        policy: RoutingPolicy | None,
        now: datetime,
    ) -> tuple[CandidateEligibility, ...]:
        return tuple(
            CandidateEligibility(candidate, self._reasons(operation, candidate, policy, now))
            for candidate in candidates
        )

    def admit(
        self,
        operation: ModelOperation,
        candidates: Sequence[ModelCandidate],
        policy: RoutingPolicy | None,
        now: datetime,
    ) -> ModelAdmission:
        eligibility = self.evaluate(operation, candidates, policy, now)
        eligible = sorted(
            (item.candidate for item in eligibility if item.eligible),
            key=lambda candidate: (candidate.cost_estimate, -candidate.priority, candidate.model_id),
        )
        if not eligible:
            reasons = {reason for item in eligibility for reason in item.reasons}
            failure = (
                AdmissionFailure.DEADLINE_INELIGIBLE
                if reasons and reasons == {EligibilityReason.DEADLINE_INELIGIBLE}
                else AdmissionFailure.NO_ELIGIBLE_MODEL
            )
            return ModelAdmission(None, None, failure, eligibility)

        capacity_admission = self._capacity.acquire_first(operation, eligible, now)
        if capacity_admission is None:
            return ModelAdmission(None, None, AdmissionFailure.CAPACITY_UNAVAILABLE, eligibility)
        return ModelAdmission(
            capacity_admission.candidate,
            capacity_admission.lease,
            None,
            eligibility,
        )

    def enqueue(
        self,
        operation: ModelOperation,
        candidates: Sequence[ModelCandidate],
        policy: RoutingPolicy | None,
    ) -> None:
        if any(item.operation.operation_id == operation.operation_id for item in self._queue):
            raise ValueError(f"operation {operation.operation_id!r} is already queued")
        self._queue.append(_QueuedOperation(self._next_sequence, operation, tuple(candidates), policy))
        self._next_sequence += 1

    def schedule_next(self, now: datetime) -> ModelAdmission | None:
        for queued in self._ordered_queue():
            decision = self.admit(queued.operation, queued.candidates, queued.policy, now)
            if decision.admitted:
                self._queue.remove(queued)
                self._last_tenant = queued.operation.tenant_id
                return decision
        return None

    def complete(self, lease: CapacityLease) -> bool:
        return self._capacity.release(lease)

    def _reasons(
        self,
        operation: ModelOperation,
        candidate: ModelCandidate,
        policy: RoutingPolicy | None,
        now: datetime,
    ) -> tuple[EligibilityReason, ...]:
        reasons: list[EligibilityReason] = []
        if policy is None or not policy.available:
            reasons.append(EligibilityReason.POLICY_UNAVAILABLE)
        elif not policy.valid:
            reasons.append(EligibilityReason.POLICY_INVALID)
        elif policy.decision is not PolicyDecision.ALLOW:
            reasons.append(EligibilityReason.POLICY_DENIED)
        else:
            if candidate.provider_id not in policy.allowed_provider_ids:
                reasons.append(EligibilityReason.PROVIDER_DENIED)
            if not operation.data_classes.issubset(policy.allowed_data_classes) or not operation.data_classes.issubset(
                candidate.allowed_data_classes
            ):
                reasons.append(EligibilityReason.DATA_POLICY_DENIED)
            if not operation.work.required_capabilities.issubset(candidate.capabilities):
                reasons.append(EligibilityReason.CAPABILITY_MISSING)
            if candidate.quality_score < max(operation.minimum_quality, policy.minimum_quality):
                reasons.append(EligibilityReason.QUALITY_INELIGIBLE)
            if not self._budget.is_available(operation, candidate):
                reasons.append(EligibilityReason.BUDGET_INELIGIBLE)
            predicted_completion = now + timedelta(
                milliseconds=candidate.predicted_latency_ms + operation.completion_reserve_ms
            )
            if operation.deadline is not None and predicted_completion > operation.deadline:
                reasons.append(EligibilityReason.DEADLINE_INELIGIBLE)
        return tuple(reasons)

    def _ordered_queue(self) -> tuple[_QueuedOperation, ...]:
        queued = sorted(self._queue, key=lambda item: item.sequence)
        if self._scheduling_order is SchedulingOrder.FIFO:
            return tuple(queued)
        if self._scheduling_order is SchedulingOrder.INTERACTIVE_FIRST:
            return tuple(
                sorted(
                    queued,
                    key=lambda item: (
                        item.operation.path is not ExecutionPath.INTERACTIVE,
                        item.sequence,
                    ),
                )
            )

        tenant_order = list(dict.fromkeys(item.operation.tenant_id for item in queued))
        if self._last_tenant in tenant_order:
            pivot = tenant_order.index(self._last_tenant) + 1
            tenant_order = tenant_order[pivot:] + tenant_order[:pivot]
        per_tenant = {
            tenant_id: [item for item in queued if item.operation.tenant_id == tenant_id]
            for tenant_id in tenant_order
        }
        ordered: list[_QueuedOperation] = []
        while any(per_tenant.values()):
            for tenant_id in tenant_order:
                if per_tenant[tenant_id]:
                    ordered.append(per_tenant[tenant_id].pop(0))
        return tuple(ordered)
