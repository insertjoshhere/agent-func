"""Validated identifiers shared by every execution path."""

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class ExecutionId:
    value: str

    def __post_init__(self) -> None:
        if not self.value.strip():
            raise ValueError("execution_id must not be blank")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, order=True)
class CorrelationId:
    value: str

    def __post_init__(self) -> None:
        if not self.value.strip():
            raise ValueError("correlation_id must not be blank")

    def __str__(self) -> str:
        return self.value
