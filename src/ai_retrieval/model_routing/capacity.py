"""Atomic hierarchical concurrency and rolling RPM/TPM admission."""

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from threading import RLock
from typing import Iterable, Mapping, Sequence

from ai_retrieval.domain.execution import ExecutionPath
from ai_retrieval.domain.model_routing import ModelCandidate, ModelOperation


class CapacityScopeKind(StrEnum):
    SYSTEM = "system"
    REQUEST = "request"
    JOB = "job"
    TENANT = "tenant"
    PROVIDER = "provider"
    MODEL = "model"


@dataclass(frozen=True, order=True)
class CapacityScope:
    kind: CapacityScopeKind
    identifier: str

    def __post_init__(self) -> None:
        if not self.identifier.strip():
            raise ValueError("capacity scope identifier must not be blank")


@dataclass(frozen=True)
class CapacityLimit:
    concurrency: int | None = None
    requests_per_minute: int | None = None
    tokens_per_minute: int | None = None

    def __post_init__(self) -> None:
        values = (self.concurrency, self.requests_per_minute, self.tokens_per_minute)
        if any(value is not None and value < 0 for value in values):
            raise ValueError("capacity limits must be non-negative")
        if all(value is None for value in values):
            raise ValueError("at least one capacity limit is required")


@dataclass(frozen=True)
class CapacityLease:
    operation_id: str
    scopes: tuple[CapacityScope, ...]


@dataclass(frozen=True)
class CapacityAdmission:
    candidate: ModelCandidate
    lease: CapacityLease


class HierarchicalCapacity:
    """Applies all configured scopes atomically; completion releases concurrency only."""

    _WINDOW_SECONDS = 60.0

    def __init__(self, limits: Mapping[CapacityScope, CapacityLimit]) -> None:
        self._limits = dict(limits)
        self._active: dict[CapacityScope, int] = defaultdict(int)
        self._events: dict[CapacityScope, deque[tuple[float, int]]] = defaultdict(deque)
        self._leases: dict[str, CapacityLease] = {}
        self._lock = RLock()

    @staticmethod
    def scopes_for(operation: ModelOperation, candidate: ModelCandidate) -> tuple[CapacityScope, ...]:
        execution_kind = (
            CapacityScopeKind.REQUEST if operation.path is ExecutionPath.INTERACTIVE else CapacityScopeKind.JOB
        )
        return (
            CapacityScope(CapacityScopeKind.SYSTEM, "system"),
            CapacityScope(execution_kind, operation.scope_id),
            CapacityScope(CapacityScopeKind.TENANT, operation.tenant_id),
            CapacityScope(CapacityScopeKind.PROVIDER, candidate.provider_id),
            CapacityScope(CapacityScopeKind.MODEL, candidate.model_id),
        )

    def acquire_first(
        self,
        operation: ModelOperation,
        ordered_candidates: Sequence[ModelCandidate],
        at: datetime,
    ) -> CapacityAdmission | None:
        timestamp = at.timestamp()
        with self._lock:
            if operation.operation_id in self._leases:
                return None
            for candidate in ordered_candidates:
                scopes = tuple(scope for scope in self.scopes_for(operation, candidate) if scope in self._limits)
                self._prune(scopes, timestamp)
                if self._fits(scopes, operation.estimated_total_tokens):
                    lease = CapacityLease(operation.operation_id, scopes)
                    for scope in scopes:
                        self._active[scope] += 1
                        self._events[scope].append((timestamp, operation.estimated_total_tokens))
                    self._leases[operation.operation_id] = lease
                    return CapacityAdmission(candidate, lease)
        return None

    def release(self, lease: CapacityLease) -> bool:
        with self._lock:
            current = self._leases.get(lease.operation_id)
            if current != lease:
                return False
            for scope in lease.scopes:
                self._active[scope] -= 1
            del self._leases[lease.operation_id]
            return True

    def active_count(self, scope: CapacityScope) -> int:
        with self._lock:
            return self._active[scope]

    def _prune(self, scopes: Iterable[CapacityScope], timestamp: float) -> None:
        cutoff = timestamp - self._WINDOW_SECONDS
        for scope in scopes:
            events = self._events[scope]
            while events and events[0][0] <= cutoff:
                events.popleft()

    def _fits(self, scopes: Iterable[CapacityScope], tokens: int) -> bool:
        for scope in scopes:
            limit = self._limits[scope]
            events = self._events[scope]
            if limit.concurrency is not None and self._active[scope] >= limit.concurrency:
                return False
            if limit.requests_per_minute is not None and len(events) >= limit.requests_per_minute:
                return False
            if limit.tokens_per_minute is not None and sum(event[1] for event in events) + tokens > limit.tokens_per_minute:
                return False
        return True
