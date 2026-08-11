"""Adapters composing current routing, security, budget, and telemetry components."""

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone

from ai_retrieval.control_plane.budget import InMemoryBudgetController
from ai_retrieval.domain.budget import Usage
from ai_retrieval.domain.execution import ExecutionContext, ExecutionPath
from ai_retrieval.domain.immutable import FrozenMapping, freeze
from ai_retrieval.domain.model_routing import ModelCandidate, ModelOperation, RoutingPolicy
from ai_retrieval.domain.work import ModelWork
from ai_retrieval.interactive.models import (
    InteractiveExecutionMetrics,
    InteractiveResponse,
    ModelInvocationResult,
    ModelPlan,
    ModelPlanningError,
)
from ai_retrieval.model_routing import AdmissionFailure, ModelRouter
from ai_retrieval.observability import ObservabilityService, TelemetryEvent
from ai_retrieval.relational.models import NormalizedResult
from ai_retrieval.security import ProtectedModelInvoker, RedactionPolicy, SecurityPolicyRegistry


class CandidateCatalog:
    """Replaceable immutable model-candidate and routing-policy lookup."""

    def __init__(
        self,
        candidates: Callable[[ModelWork, ExecutionContext], Sequence[ModelCandidate]],
        policy: Callable[[ExecutionContext], RoutingPolicy | None],
    ) -> None:
        self._candidates = candidates
        self._policy = policy

    def candidates_for(self, work: ModelWork, context: ExecutionContext) -> tuple[ModelCandidate, ...]:
        return tuple(self._candidates(work, context))

    def policy_for(self, context: ExecutionContext) -> RoutingPolicy | None:
        return self._policy(context)


class RoutedModelPlanner:
    def __init__(self, router: ModelRouter, catalog: CandidateCatalog, clock=None) -> None:
        self._router, self._catalog = router, catalog
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def plan(self, operation_id, work, context, response_reserve_ms):
        configured_input_tokens = _configuration_int(context, "estimated_model_tokens", 0)
        configured_output_tokens = _configuration_int(context, "estimated_output_tokens", 0)
        input_tokens = (
            work.estimated_input_tokens
            if work.estimated_input_tokens is not None
            else configured_input_tokens
        )
        output_tokens = (
            work.estimated_output_tokens
            if work.estimated_output_tokens is not None
            else configured_output_tokens
        )
        task_estimates_present = (
            work.estimated_input_tokens is not None
            or work.estimated_output_tokens is not None
        )
        routing_tokens = (
            input_tokens + output_tokens
            if task_estimates_present
            else configured_input_tokens
        )
        operation = ModelOperation(
            operation_id, _configuration_string(context, "tenant_id", "interactive"),
            str(context.execution_id), context.path, work,
            routing_tokens,
            minimum_quality=float(_configuration_value(context, "minimum_model_quality", 0.0)),
            data_classes=frozenset(_configuration_sequence(context, "data_classes")),
            deadline=context.timing.deadline, completion_reserve_ms=response_reserve_ms,
        )
        admission = self._router.admit(
            operation, self._catalog.candidates_for(work, context),
            self._catalog.policy_for(context), self._clock(),
        )

        if not admission.admitted:
            reason = admission.failure.value if admission.failure else "no_eligible_model"
            raise ModelPlanningError(reason)
        assert admission.candidate is not None and admission.lease is not None
        estimate = Usage(
            admission.candidate.cost_estimate,
            input_tokens,
            output_tokens,
        )
        return ModelPlan(operation, admission.candidate, admission.lease, estimate)

    def complete(self, plan: ModelPlan) -> None:
        self._router.complete(plan.lease)


class CurrentProtectedModelExecutor:
    def __init__(
        self,
        invoker: ProtectedModelInvoker,
        payload_factory: Callable[[ModelPlan, NormalizedResult], Mapping[str, object]],
        cancellation: Callable[[str], object] | None = None,
    ) -> None:
        self._invoker, self._payload_factory, self._cancellation = invoker, payload_factory, cancellation

    async def invoke(self, plan, source, context):
        identity = _configuration_string(context, "model_service_identity", "model-service")
        value = await self._invoker.invoke(
            plan.operation, plan.candidate, self._payload_factory(plan, source), identity, context
        )
        if isinstance(value, ModelInvocationResult):
            return value
        if plan.operation.work.task_definition_version is not None:
            return ModelInvocationResult(value)
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise TypeError("model gateway output must be a sequence of records")
        return ModelInvocationResult(tuple(value))

    async def cancel(self, cancellation_token: str) -> None:
        if self._cancellation is None:
            return
        result = self._cancellation(cancellation_token)
        if hasattr(result, "__await__"):
            await result


class BoundBudgetAdapter:
    def __init__(self, controller: InMemoryBudgetController) -> None:
        self._controller = controller

    def reserve(self, context, estimate):
        return self._controller.reserve(str(context.execution_id), estimate)

    def reconcile(self, reservation_id, actual):
        return self._controller.reconcile(reservation_id, actual)


class RedactedInteractiveTelemetry:
    def __init__(
        self,
        observability: ObservabilityService,
        policies: SecurityPolicyRegistry,
        clock=None,
    ) -> None:
        self._observability, self._policies = observability, policies
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    async def terminal(self, response, metrics, context):
        details = freeze({
            "request_id": response.request_id,
            "end_to_end_latency_ms": metrics.end_to_end_latency_ms,
            "deadline_outcome": response.terminal_reason.value,
            "fallback_outcome": not response.complete,
            "cancellation_duration_ms": metrics.cancellation_duration_ms,
            "late_completion_count": metrics.late_completion_count,
            "database_operation_duration_ms": metrics.database_duration_ms,
            "model_operation_duration_ms": metrics.model_duration_ms,
            "token_usage": metrics.token_usage,
            "monetary_cost_minor_units": metrics.cost_minor_units,
        })
        assert isinstance(details, FrozenMapping)
        await self._observability.emit(
            TelemetryEvent(
                str(context.correlation_id), self._clock(), "interactive_terminal",
                "interactive", response.terminal_reason.value,
                context.configuration.reference.version, details,
            ),
            self._redaction(context),
        )

    async def late_completion(self, operation_id, context):
        details = freeze({"operation_id": operation_id})
        assert isinstance(details, FrozenMapping)
        await self._observability.emit(
            TelemetryEvent(
                str(context.correlation_id), self._clock(), "late_completion",
                "interactive", "suppressed", context.configuration.reference.version, details,
            ), self._redaction(context),
        )

    def _redaction(self, context):
        policy = self._policies.resolve(context.configuration.security_policy_version)
        return policy.redaction_policy if policy else RedactionPolicy()


def _interactive_configuration(context: ExecutionContext) -> FrozenMapping:
    value = context.configuration.content.get("interactive")
    return value if isinstance(value, FrozenMapping) else FrozenMapping(())


def _configuration_value(context: ExecutionContext, key: str, default: object) -> object:
    return _interactive_configuration(context).get(key, default)


def _configuration_string(context: ExecutionContext, key: str, default: str) -> str:
    value = _configuration_value(context, key, default)
    return value if isinstance(value, str) and value.strip() else default


def _configuration_int(context: ExecutionContext, key: str, default: int) -> int:
    value = _configuration_value(context, key, default)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else default


def _configuration_sequence(context: ExecutionContext, key: str) -> tuple[str, ...]:
    value = _configuration_value(context, key, ())
    if not isinstance(value, (tuple, frozenset)):
        return ()
    return tuple(item for item in value if isinstance(item, str))
