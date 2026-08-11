"""Replaceable ports at the relational database boundary."""

from collections.abc import Mapping
from typing import Protocol

from ai_retrieval.domain.execution import ExecutionContext
from ai_retrieval.relational.models import (
    ApprovedEffect,
    DatabaseAccessDecision,
    EffectOutcome,
    QueryPlan,
    RawRelationalResult,
    SecurityAuditEvent,
)


class ReadRelationalAdapter(Protocol):
    @property
    def adapter_id(self) -> str: ...

    @property
    def adapter_version(self) -> str: ...

    def capabilities(self) -> frozenset[str]: ...

    async def execute_read(
        self,
        plan: QueryPlan,
        parameters: Mapping[str, object],
        access: DatabaseAccessDecision,
        context: ExecutionContext,
    ) -> RawRelationalResult: ...

    async def cancel(self, cancellation_token: str) -> None: ...


class ApprovedEffectAdapter(Protocol):
    """Mutation interface intentionally unavailable to the interactive DAL."""

    async def execute_approved_effect(
        self, effect: ApprovedEffect, context: ExecutionContext
    ) -> EffectOutcome: ...


class DatabaseAccessPolicy(Protocol):
    async def authorize_read(
        self, credential_id: str, plan: QueryPlan, context: ExecutionContext
    ) -> DatabaseAccessDecision: ...


class SecurityAuditSink(Protocol):
    async def emit(self, event: SecurityAuditEvent) -> None: ...
