"""Execution identity, correlation, deadline, and cancellation context creation."""

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Protocol
from uuid import uuid4

from ai_retrieval.domain.configuration import ExecutionConfiguration
from ai_retrieval.domain.execution import CancellationContext, DeadlineContext, ExecutionContext, ExecutionPath
from ai_retrieval.domain.identifiers import CorrelationId, ExecutionId


class ExecutionContextFactory(Protocol):
    def create(
        self,
        path: ExecutionPath,
        configuration: ExecutionConfiguration,
        deadline_seconds: float | None,
        cancellation_timeout_seconds: float,
    ) -> ExecutionContext: ...


class DefaultExecutionContextFactory:
    def __init__(
        self,
        clock: Callable[[], datetime] | None = None,
        identifier: Callable[[], str] | None = None,
    ) -> None:
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._identifier = identifier or (lambda: str(uuid4()))

    def create(
        self,
        path: ExecutionPath,
        configuration: ExecutionConfiguration,
        deadline_seconds: float | None,
        cancellation_timeout_seconds: float,
    ) -> ExecutionContext:
        accepted_at = self._clock()
        deadline = accepted_at + timedelta(seconds=deadline_seconds) if deadline_seconds is not None else None
        return ExecutionContext(
            execution_id=ExecutionId(self._identifier()),
            correlation_id=CorrelationId(self._identifier()),
            path=path,
            configuration=configuration,
            timing=DeadlineContext(accepted_at, deadline),
            cancellation=CancellationContext(self._identifier(), cancellation_timeout_seconds),
        )
