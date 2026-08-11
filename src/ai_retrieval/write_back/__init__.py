"""Exact approval gate and audited batch-only write-back execution."""

from ai_retrieval.write_back.audit import InMemoryWriteBackAuditSink
from ai_retrieval.write_back.authorization import InMemoryWriteBackAuthorization
from ai_retrieval.write_back.executor import BatchWriteBackExecutor, NullWriteBackAuditSink
from ai_retrieval.write_back.models import (
    ApprovalRecord, AuthorizationSnapshot, WriteBackAuditEvent, WriteBackAuditOutcome,
    WriteBackCommand, WriteBackExecutionResult, WriteBackExecutionStatus, WriteBackPolicy,
    WriteBackScope,
)
from ai_retrieval.write_back.ports import WriteBackAuditSink, WriteBackAuthorizationSource

__all__ = [
    "ApprovalRecord", "AuthorizationSnapshot", "BatchWriteBackExecutor",
    "InMemoryWriteBackAuditSink", "InMemoryWriteBackAuthorization", "NullWriteBackAuditSink",
    "WriteBackAuditEvent", "WriteBackAuditOutcome", "WriteBackAuditSink",
    "WriteBackAuthorizationSource", "WriteBackCommand", "WriteBackExecutionResult",
    "WriteBackExecutionStatus", "WriteBackPolicy", "WriteBackScope",
]
