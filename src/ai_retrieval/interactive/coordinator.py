"""Absolute-deadline read-only interactive coordinator."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from time import monotonic
from ai_retrieval.domain.budget import Usage
from ai_retrieval.domain.immutable import FrozenMapping, freeze
from ai_retrieval.domain.work import InteractiveRequest, ModelWork
from ai_retrieval.interactive.aggregator import PartialAggregator
from ai_retrieval.interactive.models import (
    InteractiveExecutionMetrics,
    InteractiveOperationState,
    InteractiveResponse,
    InteractiveTerminalReason,
    ModelPlan,
)
from ai_retrieval.interactive.ports import (
    InteractiveBudget,
    InteractiveTaskPipeline,
    InteractiveTelemetry,
    ModelPlanner,
    OutputValidator,
    ProtectedModelExecutor,
    ReadOnlyDataAccess,
)
from ai_retrieval.model_routing import AdmissionFailure
from ai_retrieval.relational import DataAccessRejection, QueryPlanReference
from ai_retrieval.security import SecurityRejection
from ai_retrieval.tasks import PackedRequest, PreparedTask, TaskFailure, TaskOutputFailure


@dataclass
class _RunState:
    terminal: bool = False
    late_completions: int = 0
    database_duration_ms: float = 0.0
    model_duration_ms: float = 0.0
    token_usage: int = 0
    cost_minor_units: int = 0


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)


class InteractiveCoordinator:
    """Runs one bounded read/model graph and returns one immutable terminal response."""

    def __init__(
        self,
        data_access: ReadOnlyDataAccess,
        model_planner: ModelPlanner,
        model_executor: ProtectedModelExecutor,
        budget: InteractiveBudget,
        validator: OutputValidator,
        telemetry: InteractiveTelemetry,
        *,
        response_reserve_ms: int = 50,
        clock=None,
    ) -> None:
        if response_reserve_ms < 0:
            raise ValueError("response reserve must be non-negative")
        self._data_access = data_access
        self._model_planner = model_planner
        self._model_executor = model_executor
        self._budget = budget
        self._validator = validator
        self._telemetry = telemetry
        self._response_reserve_ms = response_reserve_ms
        self._clock = clock or SystemClock()
        self._task_pipeline: InteractiveTaskPipeline | None = None
        self._background_tasks: set[asyncio.Task] = set()

    def dispatch(self, request, context):
        """Synchronous adapter for the existing admission router."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.execute(request, context))
        raise RuntimeError("dispatch cannot run inside an active event loop; await execute instead")

    async def execute(self, request, context):
        if context.timing.deadline is None:
            raise ValueError("interactive execution requires one absolute deadline")
        initial_ids = self._initial_operation_ids(request)
        aggregator = PartialAggregator(initial_ids)
        state = _RunState()
        started = self._clock.now()
        pending: dict[asyncio.Task, str] = {}
        cancellation_started = 0.0

        work_seconds = self._remaining_seconds(context.timing.deadline) - self._response_reserve_ms / 1000
        if work_seconds <= 0:
            aggregator.finalize_pending(InteractiveTerminalReason.DEADLINE_EXCEEDED)
        else:
            query_task = asyncio.create_task(self._read(request, context, state))
            pending[query_task] = self._query_id(request)
            done, _ = await asyncio.wait((query_task,), timeout=work_seconds)
            if query_task in done:
                pending.pop(query_task)
                try:
                    source = query_task.result()
                except Exception as error:
                    aggregator.omit(
                        self._query_id(request), self._reason(error),
                        self._details(error),
                    )
                    for operation_id in aggregator.pending:
                        aggregator.omit(operation_id, InteractiveTerminalReason.DEPENDENCY_FAILED)
                else:
                    if request.task is None:
                        aggregator.complete(self._query_id(request), source)
                        await self._run_models(request, source, context, aggregator, state, pending)
                    else:
                        aggregator = await self._run_task_request(
                            request, source, context, state, pending
                        )
            else:
                aggregator.finalize_pending(InteractiveTerminalReason.DEADLINE_EXCEEDED)

        if pending:
            cancellation_started = monotonic()
            state.terminal = True
            self._cancel_pending(pending, context, state)
        aggregator.finalize_pending(InteractiveTerminalReason.DEADLINE_EXCEEDED)
        emitted_at = min(self._clock.now(), context.timing.deadline)
        cancellation_ms = (monotonic() - cancellation_started) * 1000 if cancellation_started else 0.0
        metrics = InteractiveExecutionMetrics(
            max((emitted_at - started).total_seconds() * 1000, 0.0),
            state.database_duration_ms, state.model_duration_ms,
            state.token_usage, state.cost_minor_units, cancellation_ms,
            state.late_completions,
        )
        response = InteractiveResponse(
            request.request_id, aggregator.complete_result, aggregator.terminal_reason(),
            aggregator.results(), aggregator.incompleteness(), context.timing.deadline,
            emitted_at, context.configuration.reference.version, metrics,
        )
        state.terminal = True
        await self._safe_terminal_telemetry(response, metrics, context)
        return response

    async def _run_task_request(self, request, source, context, state, pending):
        preparation_id = self._task_preparation_id(request)
        if self._task_pipeline is None:
            aggregator = PartialAggregator((self._query_id(request), preparation_id))
            aggregator.complete(self._query_id(request), source)
            aggregator.omit(
                preparation_id,
                InteractiveTerminalReason.VALIDATION_FAILED,
                self._task_failure_details(_TaskPipelineUnavailable()),
            )
            return aggregator

        prepared = self._task_pipeline.prepare(request.task, source, context)
        if isinstance(prepared, TaskFailure):
            aggregator = PartialAggregator((self._query_id(request), preparation_id))
            aggregator.complete(self._query_id(request), source)
            aggregator.omit(
                preparation_id,
                InteractiveTerminalReason.VALIDATION_FAILED,
                self._task_failure_details(prepared),
            )
            return aggregator

        operation_ids = tuple(
            self._pack_id(pack) for pack in prepared.packs
        )
        aggregator = PartialAggregator((self._query_id(request), *operation_ids))
        aggregator.complete(self._query_id(request), source)
        await self._run_task_models(
            prepared, source, context, aggregator, state, pending
        )
        return aggregator

    async def _run_task_models(self, prepared, source, context, aggregator, state, pending):
        for pack, work in zip(prepared.packs, prepared.model_work, strict=True):
            operation_id = self._pack_id(pack)
            if self._remaining_seconds(context.timing.deadline) <= self._response_reserve_ms / 1000:
                aggregator.omit(operation_id, InteractiveTerminalReason.DEADLINE_EXCEEDED)
                continue
            task = asyncio.create_task(
                self._task_model(operation_id, prepared, pack, work, source, context, state)
            )
            pending[task] = operation_id
        await self._collect_models(context, aggregator, pending)

    async def _run_models(self, request, source, context, aggregator, state, pending):
        for index, work in enumerate(request.model_work):
            operation_id = self._model_id(index, work)
            if self._remaining_seconds(context.timing.deadline) <= self._response_reserve_ms / 1000:
                aggregator.omit(operation_id, InteractiveTerminalReason.DEADLINE_EXCEEDED)
                continue
            task = asyncio.create_task(self._model(operation_id, work, source, context, state))
            pending[task] = operation_id
        await self._collect_models(context, aggregator, pending)

    async def _collect_models(self, context, aggregator, pending):
        if not pending:
            return
        model_tasks = tuple(pending)
        timeout = max(self._remaining_seconds(context.timing.deadline) - self._response_reserve_ms / 1000, 0)
        done, not_done = await asyncio.wait(model_tasks, timeout=timeout)
        for task in done:
            operation_id = pending.pop(task)
            try:
                aggregator.complete(operation_id, task.result())
            except Exception as error:
                aggregator.omit(operation_id, self._reason(error), self._details(error))
        for task in not_done:
            operation_id = pending[task]
            aggregator.omit(operation_id, InteractiveTerminalReason.DEADLINE_EXCEEDED)

    async def _read(self, request, context, state):
        started = monotonic()
        try:
            version = self._query_plan_version(context)
            return await self._data_access.execute_read(
                QueryPlanReference(request.query.plan_id, version), dict(request.query.parameters), context
            )
        finally:
            state.database_duration_ms += (monotonic() - started) * 1000

    async def _task_model(self, operation_id, prepared, pack, work, source, context, state):
        plan: ModelPlan | None = None
        reservation_id: str | None = None
        actual = None
        started = monotonic()
        try:
            plan = self._model_planner.plan(operation_id, work, context, self._response_reserve_ms)
            decision = self._budget.reserve(context, plan.estimate)
            if not decision.accepted:
                raise _BudgetExhausted(decision.exhaustion.exhausted_scopes)
            reservation_id = decision.reservation.reservation_id
            invoked = await self._model_executor.invoke(plan, source, context)
            actual = invoked.actual_usage
            assert self._task_pipeline is not None
            validation = self._task_pipeline.parse_and_validate(
                prepared, pack, invoked.output, context
            )
            if isinstance(validation, TaskOutputFailure):
                raise _TaskFailed(validation)
            if not validation.outcome.accepted:
                raise _ValidationFailed(
                    validation.outcome.reason_codes,
                    validation.outcome.failed_rule_ids,
                )
            state.token_usage += actual.total_tokens
            state.cost_minor_units += actual.cost_minor_units
            return validation.accepted_records
        finally:
            if reservation_id is not None:
                self._budget.reconcile(reservation_id, actual or plan.estimate)
            if plan is not None:
                self._model_planner.complete(plan)
            state.model_duration_ms += (monotonic() - started) * 1000

    async def _model(self, operation_id, work, source, context, state):
        plan: ModelPlan | None = None
        reservation_id: str | None = None
        actual = None
        started = monotonic()
        try:
            plan = self._model_planner.plan(operation_id, work, context, self._response_reserve_ms)
            decision = self._budget.reserve(context, plan.estimate)
            if not decision.accepted:
                scopes = decision.exhaustion.exhausted_scopes
                raise _BudgetExhausted(scopes)
            reservation_id = decision.reservation.reservation_id
            invoked = await self._model_executor.invoke(plan, source, context)
            actual = invoked.actual_usage
            if not isinstance(invoked.output, tuple):
                raise TypeError("legacy model output must be a tuple of records")
            validation = self._validator.validate(work.input_ids, invoked.output, context)
            if not validation.outcome.accepted:
                raise _ValidationFailed(
                    validation.outcome.reason_codes,
                    getattr(validation.outcome, "failed_rule_ids", ()),
                )
            state.token_usage += actual.total_tokens
            state.cost_minor_units += actual.cost_minor_units
            return validation.accepted_records
        finally:
            if reservation_id is not None:
                self._budget.reconcile(reservation_id, actual or plan.estimate)
            if plan is not None:
                self._model_planner.complete(plan)
            state.model_duration_ms += (monotonic() - started) * 1000

    def _cancel_pending(self, pending, context, state):
        for task, operation_id in tuple(pending.items()):
            task.cancel()
            self._start_background(self._observe_late(task, operation_id, context, state))
        self._start_background(self._cancel_dependencies(context))

    async def _cancel_dependencies(self, context):
        calls = (
            asyncio.create_task(self._data_access.cancel(context.cancellation.token)),
            asyncio.create_task(self._model_executor.cancel(context.cancellation.token)),
        )
        if context.cancellation.timeout_seconds == 0:
            await asyncio.sleep(0)
            pending = tuple(call for call in calls if not call.done())
        else:
            _, pending = await asyncio.wait(calls, timeout=context.cancellation.timeout_seconds)
        for call in pending:
            call.cancel()
        await asyncio.gather(*calls, return_exceptions=True)

    async def _observe_late(self, task, operation_id, context, state):
        try:
            await task
        except asyncio.CancelledError:
            return
        except BaseException:
            pass
        state.late_completions += 1
        await self._safe_late_telemetry(operation_id, context)

    def _start_background(self, coroutine):
        task = asyncio.create_task(coroutine)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_done)

    def _background_done(self, task):
        self._background_tasks.discard(task)
        if task.cancelled():
            return
        try:
            task.result()
        except BaseException:
            pass

    async def _safe_terminal_telemetry(self, response, metrics, context):
        try:
            await self._telemetry.terminal(response, metrics, context)
        except Exception:
            pass

    async def _safe_late_telemetry(self, operation_id, context):
        try:
            await self._telemetry.late_completion(operation_id, context)
        except Exception:
            pass

    def _remaining_seconds(self, deadline):
        return max((deadline - self._clock.now()).total_seconds(), 0.0)

    @staticmethod
    def _initial_operation_ids(request):
        if request.task is not None:
            return (
                InteractiveCoordinator._query_id(request),
                InteractiveCoordinator._task_preparation_id(request),
            )
        return (InteractiveCoordinator._query_id(request),) + tuple(
            InteractiveCoordinator._model_id(index, work)
            for index, work in enumerate(request.model_work)
        )

    @staticmethod
    def _task_preparation_id(request):
        return f"task:{request.task.function}:prepare"

    @staticmethod
    def _pack_id(pack: PackedRequest):
        return f"task:{pack.function.value}:{pack.definition_version}:pack:{pack.pack_index}"

    @staticmethod
    def _query_id(request):
        return f"query:{request.query.plan_id}"

    @staticmethod
    def _model_id(index, work):
        return f"model:{index}:{work.task_type}"

    @staticmethod
    def _query_plan_version(context):
        database = context.configuration.content.get("database")
        version = database.get("query_plan_version") if isinstance(database, FrozenMapping) else None
        return version if isinstance(version, str) and version.strip() else "1"

    @staticmethod
    def _task_failure_details(failure):
        if isinstance(failure, _TaskPipelineUnavailable):
            return freeze({
                "failure_code": "task_pipeline_unavailable",
                "failure_stage": "preparation",
                "failed_rule_ids": ("interactive_task.pipeline",),
            })
        return freeze({
            "failure_code": failure.code.value,
            "failure_stage": failure.stage.value,
            "failed_rule_ids": failure.failed_rule_ids,
            "failure_details": failure.details,
        })

    @staticmethod
    def _details(error):
        if isinstance(error, _BudgetExhausted):
            value = freeze({"exhausted_scopes": error.scopes})
        elif isinstance(error, _TaskFailed):
            value = InteractiveCoordinator._task_failure_details(error.failure)
        elif isinstance(error, _ValidationFailed):
            value = freeze({
                "reason_codes": error.reasons,
                "failed_rule_ids": error.failed_rule_ids,
            })
        else:
            failure = getattr(error, "failure", None)
            code = getattr(failure, "code", None)
            value = freeze({"reason_code": getattr(code, "value", type(error).__name__)})
        assert isinstance(value, FrozenMapping)
        return value

    @staticmethod
    def _reason(error):
        if isinstance(error, _BudgetExhausted):
            return InteractiveTerminalReason.BUDGET_EXHAUSTED
        if isinstance(error, _ValidationFailed) or isinstance(error, _TaskFailed):
            return InteractiveTerminalReason.VALIDATION_FAILED
        if isinstance(error, SecurityRejection):
            return InteractiveTerminalReason.SECURITY_REJECTED
        if isinstance(error, DataAccessRejection):
            return InteractiveTerminalReason.DATABASE_FAILURE
        reason = getattr(error, "reason", None)
        if reason == AdmissionFailure.DEADLINE_INELIGIBLE.value:
            return InteractiveTerminalReason.DEADLINE_INELIGIBLE
        if isinstance(error, asyncio.CancelledError):
            return InteractiveTerminalReason.CANCELLED
        return InteractiveTerminalReason.MODEL_FAILURE


class TaskAwareInteractiveCoordinator(InteractiveCoordinator):
    """Interactive coordinator variant with only a read-only task pipeline."""

    def __init__(
        self,
        data_access: ReadOnlyDataAccess,
        model_planner: ModelPlanner,
        model_executor: ProtectedModelExecutor,
        budget: InteractiveBudget,
        validator: OutputValidator,
        telemetry: InteractiveTelemetry,
        task_pipeline: InteractiveTaskPipeline,
        *,
        response_reserve_ms: int = 50,
        clock=None,
    ) -> None:
        super().__init__(
            data_access, model_planner, model_executor, budget, validator, telemetry,
            response_reserve_ms=response_reserve_ms, clock=clock,
        )
        self._task_pipeline = task_pipeline


class _BudgetExhausted(RuntimeError):
    def __init__(self, scopes):
        super().__init__("budget exhausted")
        self.scopes = tuple(scopes)


class _ValidationFailed(RuntimeError):
    def __init__(self, reasons, failed_rule_ids=()):
        super().__init__("validation failed")
        self.reasons = tuple(reasons)
        self.failed_rule_ids = tuple(failed_rule_ids)


class _TaskFailed(RuntimeError):
    def __init__(self, failure: TaskOutputFailure):
        super().__init__(failure.code.value)
        self.failure = failure


class _TaskPipelineUnavailable(RuntimeError):
    pass
