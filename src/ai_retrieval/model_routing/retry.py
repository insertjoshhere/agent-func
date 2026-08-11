"""Pure bounded retry and configured fallback state transitions."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Mapping

from ai_retrieval.domain.model_routing import ModelCandidate, ModelOperation


@dataclass(frozen=True)
class RetryPolicy:
    retryable_failure_classes: frozenset[str]
    max_attempts: int
    delay_schedule_ms: tuple[int, ...]
    fallback_model_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        if any(delay < 0 for delay in self.delay_schedule_ms):
            raise ValueError("retry delays must be non-negative")
        if any(not model_id.strip() for model_id in self.fallback_model_ids):
            raise ValueError("fallback model identifiers must not be blank")


@dataclass(frozen=True)
class AttemptState:
    current_model_id: str
    attempts_completed: int

    def __post_init__(self) -> None:
        if not self.current_model_id.strip() or self.attempts_completed < 1:
            raise ValueError("attempt state requires a model and at least one completed attempt")


@dataclass(frozen=True)
class RetryGateSnapshot:
    budget_available: bool
    capacity_available: bool
    model_eligible: bool


class RetryTransitionKind(StrEnum):
    RETRY = "retry"
    FALLBACK = "fallback"
    TERMINAL = "terminal"


class RetryTerminalReason(StrEnum):
    NONRETRYABLE_FAILURE = "nonretryable_failure"
    ATTEMPTS_EXHAUSTED = "attempts_exhausted"
    DELAY_NOT_CONFIGURED = "delay_not_configured"
    MODEL_INELIGIBLE = "model_ineligible"
    BUDGET_INELIGIBLE = "budget_ineligible"
    CAPACITY_UNAVAILABLE = "capacity_unavailable"
    DEADLINE_INELIGIBLE = "deadline_ineligible"


@dataclass(frozen=True)
class RetryTransition:
    kind: RetryTransitionKind
    model_id: str | None = None
    delay_ms: int | None = None
    terminal_reason: RetryTerminalReason | None = None

    @property
    def invokes_model(self) -> bool:
        return self.kind is not RetryTransitionKind.TERMINAL


class RetryController:
    def next_transition(
        self,
        operation: ModelOperation,
        state: AttemptState,
        failure_class: str,
        policy: RetryPolicy,
        candidates: Mapping[str, ModelCandidate],
        gates: RetryGateSnapshot,
        now: datetime,
    ) -> RetryTransition:
        if failure_class not in policy.retryable_failure_classes:
            return self._terminal(RetryTerminalReason.NONRETRYABLE_FAILURE)
        if state.attempts_completed >= policy.max_attempts:
            return self._terminal(RetryTerminalReason.ATTEMPTS_EXHAUSTED)

        retry_index = state.attempts_completed - 1
        if retry_index >= len(policy.delay_schedule_ms):
            return self._terminal(RetryTerminalReason.DELAY_NOT_CONFIGURED)
        delay_ms = policy.delay_schedule_ms[retry_index]
        model_id = (
            policy.fallback_model_ids[retry_index]
            if retry_index < len(policy.fallback_model_ids)
            else state.current_model_id
        )
        candidate = candidates.get(model_id)
        if candidate is None or not gates.model_eligible:
            return self._terminal(RetryTerminalReason.MODEL_INELIGIBLE)
        if not gates.budget_available:
            return self._terminal(RetryTerminalReason.BUDGET_INELIGIBLE)
        if not gates.capacity_available:
            return self._terminal(RetryTerminalReason.CAPACITY_UNAVAILABLE)

        completion = now + timedelta(
            milliseconds=delay_ms + candidate.predicted_latency_ms + operation.completion_reserve_ms
        )
        if operation.deadline is not None and completion > operation.deadline:
            return self._terminal(RetryTerminalReason.DEADLINE_INELIGIBLE)

        kind = (
            RetryTransitionKind.RETRY
            if model_id == state.current_model_id
            else RetryTransitionKind.FALLBACK
        )
        return RetryTransition(kind, model_id, delay_ms, None)

    @staticmethod
    def _terminal(reason: RetryTerminalReason) -> RetryTransition:
        return RetryTransition(RetryTransitionKind.TERMINAL, terminal_reason=reason)
