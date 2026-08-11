"""Replaceable ports available only to the bulk write-back component."""

from typing import Protocol

from ai_retrieval.write_back.models import AuthorizationSnapshot, WriteBackAuditEvent


class WriteBackAuthorizationSource(Protocol):
    def resolve(self, policy_version: str, approval_reference: str) -> AuthorizationSnapshot: ...


class WriteBackAuditSink(Protocol):
    """Append-only sink; implementations must not mutate prior audit records."""

    def append(self, event: WriteBackAuditEvent) -> None: ...
