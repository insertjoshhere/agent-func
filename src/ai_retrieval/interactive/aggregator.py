"""Deterministic partial aggregation with exact incompleteness metadata."""

from ai_retrieval.domain.immutable import FrozenMapping
from ai_retrieval.interactive.models import (
    InteractiveOperationResult,
    InteractiveOperationState,
    InteractiveTerminalReason,
    OperationIncompleteness,
)


_REASON_PRIORITY = {
    InteractiveTerminalReason.DEADLINE_EXCEEDED: 0,
    InteractiveTerminalReason.BUDGET_EXHAUSTED: 1,
    InteractiveTerminalReason.SECURITY_REJECTED: 2,
    InteractiveTerminalReason.DATABASE_FAILURE: 3,
    InteractiveTerminalReason.VALIDATION_FAILED: 4,
    InteractiveTerminalReason.DEADLINE_INELIGIBLE: 5,
    InteractiveTerminalReason.MODEL_FAILURE: 6,
    InteractiveTerminalReason.CANCELLED: 7,
    InteractiveTerminalReason.DEPENDENCY_FAILED: 8,
    InteractiveTerminalReason.INTERNAL_FAILURE: 9,
}


class PartialAggregator:
    """Collect each requested operation at most once and finalize pending work exactly."""

    def __init__(self, operation_ids: tuple[str, ...]) -> None:
        if not operation_ids or len(set(operation_ids)) != len(operation_ids):
            raise ValueError("aggregation requires unique requested operations")
        self._operation_ids = operation_ids
        self._results: dict[str, object] = {}
        self._incomplete: dict[str, OperationIncompleteness] = {}

    def complete(self, operation_id: str, value: object) -> None:
        self._require_pending(operation_id)
        self._results[operation_id] = value

    def omit(
        self,
        operation_id: str,
        reason: InteractiveTerminalReason,
        details: FrozenMapping = FrozenMapping(()),
        state: InteractiveOperationState = InteractiveOperationState.OMITTED,
    ) -> None:
        self._require_pending(operation_id)
        self._incomplete[operation_id] = OperationIncompleteness(operation_id, state, reason, details)

    def finalize_pending(self, reason: InteractiveTerminalReason) -> None:
        for operation_id in self.pending:
            self.omit(operation_id, reason, state=InteractiveOperationState.INCOMPLETE)

    @property
    def pending(self) -> tuple[str, ...]:
        return tuple(
            operation_id for operation_id in self._operation_ids
            if operation_id not in self._results and operation_id not in self._incomplete
        )


    @property
    def complete_result(self) -> bool:
        return not self.pending and not self._incomplete

    def results(self) -> tuple[InteractiveOperationResult, ...]:
        return tuple(
            InteractiveOperationResult(operation_id, self._results[operation_id])
            for operation_id in self._operation_ids if operation_id in self._results
        )

    def incompleteness(self) -> tuple[OperationIncompleteness, ...]:
        return tuple(
            self._incomplete[operation_id]
            for operation_id in self._operation_ids if operation_id in self._incomplete
        )

    def terminal_reason(self) -> InteractiveTerminalReason:
        if not self._incomplete:
            return InteractiveTerminalReason.COMPLETE
        return min(
            (item.reason for item in self._incomplete.values()),
            key=lambda reason: _REASON_PRIORITY[reason],
        )

    def _require_pending(self, operation_id: str) -> None:
        if operation_id not in self._operation_ids:
            raise KeyError(f"unknown operation: {operation_id}")
        if operation_id in self._results or operation_id in self._incomplete:
            raise ValueError(f"operation already terminal: {operation_id}")
