"""Replaceable read-only ports used by the interactive coordinator."""

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Protocol, TypeVar

from ai_retrieval.domain.budget import ReconciliationReceipt, ReservationDecision, Usage
from ai_retrieval.domain.execution import ExecutionContext
from ai_retrieval.domain.model_routing import ModelCandidate, ModelOperation
from ai_retrieval.domain.work import InteractiveRequest, InteractiveTaskWork, ModelWork
from ai_retrieval.interactive.models import (
    InteractiveExecutionMetrics,
    InteractiveResponse,
    ModelInvocationResult,
    ModelPlan,
)
from ai_retrieval.relational.models import NormalizedResult, QueryPlanReference
from ai_retrieval.tasks.models import PackedRequest, PreparedTask, TaskFailure, TaskOutputFailure
from ai_retrieval.validation.models import ValidationResult


T_co = TypeVar("T_co", covariant=True)


class InteractiveDispatcher(Protocol[T_co]):
    def dispatch(self, request: InteractiveRequest, context: ExecutionContext) -> T_co: ...


class Clock(Protocol):
    def now(self) -> datetime: ...


class ReadOnlyDataAccess(Protocol):
    async def execute_read(
        self, reference: QueryPlanReference, parameters: Mapping[str, object], context: ExecutionContext
    ) -> NormalizedResult: ...

    async def cancel(self, cancellation_token: str) -> None: ...


class ModelPlanner(Protocol):
    def plan(
        self, operation_id: str, work: ModelWork, context: ExecutionContext, response_reserve_ms: int
    ) -> ModelPlan: ...

    def complete(self, plan: ModelPlan) -> None: ...


class ProtectedModelExecutor(Protocol):
    async def invoke(
        self, plan: ModelPlan, source: NormalizedResult, context: ExecutionContext
    ) -> ModelInvocationResult: ...

    async def cancel(self, cancellation_token: str) -> None: ...


class InteractiveBudget(Protocol):
    def reserve(self, context: ExecutionContext, estimate: Usage) -> ReservationDecision: ...

    def reconcile(self, reservation_id: str, actual: Usage) -> ReconciliationReceipt: ...


class OutputValidator(Protocol):
    def validate(
        self,
        input_ids: Sequence[str],
        output: Sequence[Mapping[str, object]],
        context: ExecutionContext,
    ) -> ValidationResult: ...


class InteractiveTaskPipeline(Protocol):
    def prepare(
        self,
        selection: InteractiveTaskWork,
        source: NormalizedResult,
        context: ExecutionContext,
    ) -> PreparedTask | TaskFailure: ...

    def parse_and_validate(
        self,
        prepared: PreparedTask,
        pack: PackedRequest,
        response: object,
        context: ExecutionContext,
    ) -> ValidationResult | TaskOutputFailure: ...


class InteractiveTelemetry(Protocol):
    async def terminal(
        self, response: InteractiveResponse, metrics: InteractiveExecutionMetrics,
        context: ExecutionContext,
    ) -> None: ...

    async def late_completion(
        self, operation_id: str, context: ExecutionContext
    ) -> None: ...
