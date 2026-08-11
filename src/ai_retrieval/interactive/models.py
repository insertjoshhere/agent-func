"""Immutable outcomes for deadline-bounded interactive execution."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from ai_retrieval.domain.budget import Usage
from ai_retrieval.domain.immutable import FrozenMapping
from ai_retrieval.domain.model_routing import ModelCandidate, ModelOperation


class InteractiveTerminalReason(StrEnum):
    COMPLETE = "complete"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    DEADLINE_INELIGIBLE = "deadline_ineligible"
    BUDGET_EXHAUSTED = "budget_exhausted"
    DATABASE_FAILURE = "database_failure"
    MODEL_FAILURE = "model_failure"
    VALIDATION_FAILED = "validation_failed"
    SECURITY_REJECTED = "security_rejected"
    CANCELLED = "cancelled"
    INTERNAL_FAILURE = "internal_failure"
    DEPENDENCY_FAILED = "dependency_failed"


class InteractiveOperationState(StrEnum):
    COMPLETED = "completed"
    OMITTED = "omitted"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True)
class OperationIncompleteness:
    operation_id: str
    state: InteractiveOperationState
    reason: InteractiveTerminalReason
    details: FrozenMapping = field(default_factory=lambda: FrozenMapping(()))

    def __post_init__(self) -> None:
        if not self.operation_id.strip() or self.state is InteractiveOperationState.COMPLETED:
            raise ValueError("incompleteness requires a non-completed operation")


@dataclass(frozen=True)
class InteractiveOperationResult:
    operation_id: str
    value: object


@dataclass(frozen=True)
class InteractiveExecutionMetrics:
    end_to_end_latency_ms: float
    database_duration_ms: float
    model_duration_ms: float
    token_usage: int
    cost_minor_units: int
    cancellation_duration_ms: float
    late_completion_count: int


@dataclass(frozen=True)
class InteractiveResponse:
    request_id: str
    complete: bool
    terminal_reason: InteractiveTerminalReason
    results: tuple[InteractiveOperationResult, ...]
    incompleteness: tuple[OperationIncompleteness, ...]
    deadline: datetime
    emitted_at: datetime
    configuration_version: str
    metrics: InteractiveExecutionMetrics

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("response request_id must not be blank")
        if self.complete != (not self.incompleteness):
            raise ValueError("only responses without incompleteness are complete")
        if self.complete != (self.terminal_reason is InteractiveTerminalReason.COMPLETE):
            raise ValueError("complete responses require the complete terminal reason")
        included = {item.operation_id for item in self.results}
        omitted = {item.operation_id for item in self.incompleteness}
        if included & omitted:
            raise ValueError("an operation cannot be included and incomplete")


@dataclass(frozen=True)
class ModelPlan:
    operation: ModelOperation
    candidate: ModelCandidate
    lease: object
    estimate: Usage


@dataclass(frozen=True)
class ModelInvocationResult:
    output: object
    actual_usage: Usage = Usage()


class ModelPlanningError(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason
