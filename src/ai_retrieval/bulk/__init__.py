"""Durable bulk-path contracts and in-memory prototype adapters."""

from ai_retrieval.bulk.coordinator import BulkCoordinator
from ai_retrieval.bulk.execution import (
    BulkWorkExecutor,
    NullBulkTelemetrySink,
    build_terminal_report,
    classify_terminal_job,
)
from ai_retrieval.bulk.effects import EffectConflictError, EffectRecoveryCoordinator, EffectRecoveryError
from ai_retrieval.bulk.effect_models import (
    EffectAttempt, EffectAttemptStatus, EffectEvidence, EffectEvidenceStatus,
    EffectRecord, EffectRecoveryResult, EffectRecoveryStatus, EffectRequest,
    ReconciliationRecord, TransactionBoundary,
)
from ai_retrieval.bulk.memory import (
    CheckpointTransitionError,
    DeadLetterPersistenceError,
    InMemoryDeadLetterQueue,
    InMemoryDurableWorkRepository,
    InMemoryNotificationBroker,
    InMemoryObjectResultStore,
    LeaseOwnershipError,
    WorkConflictError,
    WorkNotFoundError,
)
from ai_retrieval.bulk.models import (
    ArtifactKind,
    BrokerDelivery,
    BulkStateTelemetry,
    CheckpointRecord,
    CheckpointStage,
    ClaimStatus,
    DeadLetterEntry,
    JobTerminalCause,
    ObjectReference,
    OutboxRecord,
    OutboxStatus,
    PersistenceFailureOutcome,
    ResumeState,
    TerminalJobClassification,
    TerminalJobReport,
    TerminalStateGroup,
    TerminalWorkItemRecord,
    TerminalWorkItemState,
    WorkClaim,
    WorkItemExecutionResult,
    WorkItemRecord,
    WorkNotification,
    WorkState,
    WorkSubmission,
)
from ai_retrieval.bulk.ports import (
    BulkDispatcher, DurableWorkRepository, EffectAdapter, EffectRecoveryRepository,
    NotificationBroker, ObjectResultStore,
)
from ai_retrieval.bulk.worker import BulkStageAction, BulkStageResult, BulkWorker

__all__ = [
    "ArtifactKind", "BrokerDelivery", "BulkCoordinator", "BulkDispatcher",
    "BulkStageAction", "BulkStageResult", "BulkStateTelemetry", "BulkWorker",
    "BulkWorkExecutor", "CheckpointRecord", "CheckpointStage",
    "CheckpointTransitionError", "ClaimStatus", "DeadLetterEntry",
    "DeadLetterPersistenceError", "DurableWorkRepository", "EffectAdapter", "EffectAttempt",
    "EffectAttemptStatus", "EffectConflictError", "EffectEvidence", "EffectEvidenceStatus",
    "EffectRecord", "EffectRecoveryCoordinator", "EffectRecoveryError",
    "EffectRecoveryRepository", "EffectRecoveryResult", "EffectRecoveryStatus", "EffectRequest",
    "InMemoryDeadLetterQueue", "InMemoryDurableWorkRepository", "InMemoryNotificationBroker",
    "InMemoryObjectResultStore", "JobTerminalCause", "LeaseOwnershipError",
    "NotificationBroker", "NullBulkTelemetrySink", "ObjectReference", "ObjectResultStore",
    "OutboxRecord", "OutboxStatus", "PersistenceFailureOutcome", "ReconciliationRecord",
    "ResumeState", "TerminalJobClassification", "TerminalJobReport", "TerminalStateGroup",
    "TerminalWorkItemRecord", "TerminalWorkItemState", "TransactionBoundary", "WorkClaim",
    "WorkConflictError", "WorkItemExecutionResult", "WorkItemRecord", "WorkNotFoundError",
    "WorkNotification", "WorkState", "WorkSubmission", "build_terminal_report",
    "classify_terminal_job",
]
