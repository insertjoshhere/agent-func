"""Immutable admission envelope and routing discriminator."""

from dataclasses import dataclass

from ai_retrieval.domain.configuration import ConfigurationReference
from ai_retrieval.domain.execution import ExecutionPath
from ai_retrieval.domain.work import BatchJob, InteractiveRequest


@dataclass(frozen=True)
class AdmissionEnvelope:
    configuration: ConfigurationReference
    interactive: InteractiveRequest | None = None
    bulk: BatchJob | None = None
    discriminator: ExecutionPath | None = None
    interactive_deadline_seconds: float | None = None
    cancellation_timeout_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.interactive_deadline_seconds is not None and self.interactive_deadline_seconds <= 0:
            raise ValueError("interactive deadline duration must be positive")
        if self.cancellation_timeout_seconds < 0:
            raise ValueError("cancellation timeout must be non-negative")
