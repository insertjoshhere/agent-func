"""Normalized admission and dispatch outcomes."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, TypeVar

from ai_retrieval.domain.execution import ExecutionContext, ExecutionPath
from ai_retrieval.domain.failures import TypedFailure
from ai_retrieval.domain.immutable import FrozenMapping


class OutcomeStatus(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    COMPLETED = "completed"


@dataclass(frozen=True)
class ExecutionOutcome:
    status: OutcomeStatus
    component: str
    details: FrozenMapping = FrozenMapping(())


T = TypeVar("T")


@dataclass(frozen=True)
class AdmissionDecision(Generic[T]):
    status: OutcomeStatus
    path: ExecutionPath | None
    context: ExecutionContext | None
    outcome: T | None
    failure: TypedFailure | None

    def __post_init__(self) -> None:
        accepted = self.status is OutcomeStatus.ACCEPTED
        if accepted != (self.path is not None and self.context is not None and self.outcome is not None):
            raise ValueError("accepted decisions require path, context, and outcome")
        if accepted == (self.failure is not None):
            raise ValueError("only rejected decisions require a failure")
