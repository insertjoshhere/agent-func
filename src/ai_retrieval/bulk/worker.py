"""Claim-aware durable bulk worker orchestration."""

from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum
from typing import Protocol

from ai_retrieval.bulk.models import (
    CheckpointStage, TerminalJobReport, TerminalWorkItemRecord, WorkClaim,
)
from ai_retrieval.domain.execution import ExecutionContext


class BulkStageAction(StrEnum):
    CHECKPOINTED = "checkpointed"
    TERMINAL = "terminal"


@dataclass(frozen=True)
class BulkStageResult:
    action: BulkStageAction
    checkpoint_stage: CheckpointStage | None = None
    terminal: TerminalWorkItemRecord | None = None
    outcome: str = "processed"


class BulkStageProcessor(Protocol):
    def process_stage(
        self, claim: WorkClaim, context: ExecutionContext,
    ) -> BulkStageResult: ...


class ClaimSource(Protocol):
    def claim(self, job_id: str, idempotency_key: str, owner: str, now, lease_duration: timedelta) -> WorkClaim: ...
    def resume_state(self, job_id: str, idempotency_key: str): ...


class BulkWorker:
    """Runs durable stages one claim at a time and resumes from checkpoints."""

    def __init__(
        self, repository: ClaimSource, processor: BulkStageProcessor,
        clock, lease_duration: timedelta = timedelta(minutes=5),
    ) -> None:
        if lease_duration.total_seconds() <= 0:
            raise ValueError("lease duration must be positive")
        self._repository = repository
        self._processor = processor
        self._clock = clock
        self._lease_duration = lease_duration

    def resume(
        self, job_id: str, idempotency_key: str, owner: str,
        context: ExecutionContext, *, max_stages: int | None = None,
    ) -> tuple[BulkStageResult, ...]:
        if max_stages is not None and max_stages <= 0:
            raise ValueError("max_stages must be positive")
        results: list[BulkStageResult] = []
        while max_stages is None or len(results) < max_stages:
            state = self._repository.resume_state(job_id, idempotency_key)
            if state.next_stage is None:
                break
            claim = self._repository.claim(
                job_id, idempotency_key, owner, self._clock(), self._lease_duration,
            )
            if not claim.acquired:
                break
            result = self._processor.process_stage(claim, context)
            results.append(result)
            if result.action is BulkStageAction.TERMINAL:
                break
        return tuple(results)
