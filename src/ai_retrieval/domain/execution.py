"""Execution, deadline, and cooperative cancellation context."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from ai_retrieval.domain.configuration import ExecutionConfiguration
from ai_retrieval.domain.identifiers import CorrelationId, ExecutionId


class ExecutionPath(StrEnum):
    INTERACTIVE = "interactive"
    BULK = "bulk"


@dataclass(frozen=True)
class CancellationContext:
    token: str
    timeout_seconds: float

    def __post_init__(self) -> None:
        if not self.token.strip():
            raise ValueError("cancellation token must not be blank")
        if self.timeout_seconds < 0:
            raise ValueError("cancellation timeout must be non-negative")


@dataclass(frozen=True)
class DeadlineContext:
    accepted_at: datetime
    deadline: datetime | None

    def __post_init__(self) -> None:
        if self.deadline is not None and self.deadline < self.accepted_at:
            raise ValueError("deadline must not precede acceptance")


@dataclass(frozen=True)
class ExecutionContext:
    execution_id: ExecutionId
    correlation_id: CorrelationId
    path: ExecutionPath
    configuration: ExecutionConfiguration
    timing: DeadlineContext
    cancellation: CancellationContext
