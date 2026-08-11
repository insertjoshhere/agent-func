"""Immutable model-routing, capacity, scheduling, and retry values."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from ai_retrieval.domain.execution import ExecutionPath
from ai_retrieval.domain.work import ModelWork


class PolicyDecision(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True)
class ModelCandidate:
    model_id: str
    provider_id: str
    capabilities: frozenset[str]
    allowed_data_classes: frozenset[str]
    cost_estimate: int
    priority: int
    predicted_latency_ms: int
    quality_score: float

    def __post_init__(self) -> None:
        if not self.model_id.strip() or not self.provider_id.strip():
            raise ValueError("model and provider identifiers must not be blank")
        if self.cost_estimate < 0 or self.predicted_latency_ms < 0:
            raise ValueError("model cost and predicted latency must be non-negative")
        if not 0 <= self.quality_score <= 1:
            raise ValueError("quality_score must be between zero and one")


@dataclass(frozen=True)
class ModelOperation:
    operation_id: str
    tenant_id: str
    scope_id: str
    path: ExecutionPath
    work: ModelWork
    estimated_total_tokens: int
    minimum_quality: float = 0.0
    data_classes: frozenset[str] = frozenset()
    deadline: datetime | None = None
    completion_reserve_ms: int = 0

    def __post_init__(self) -> None:
        if not self.operation_id.strip() or not self.tenant_id.strip() or not self.scope_id.strip():
            raise ValueError("operation, tenant, and scope identifiers must not be blank")
        if self.estimated_total_tokens < 0 or self.completion_reserve_ms < 0:
            raise ValueError("token estimate and completion reserve must be non-negative")
        if not 0 <= self.minimum_quality <= 1:
            raise ValueError("minimum_quality must be between zero and one")


@dataclass(frozen=True)
class RoutingPolicy:
    version: str
    decision: PolicyDecision
    available: bool
    valid: bool
    allowed_provider_ids: frozenset[str]
    allowed_data_classes: frozenset[str]
    minimum_quality: float = 0.0

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("routing policy version must not be blank")
        if not 0 <= self.minimum_quality <= 1:
            raise ValueError("policy minimum_quality must be between zero and one")

    @property
    def permits_routing(self) -> bool:
        return self.available and self.valid and self.decision is PolicyDecision.ALLOW


class EligibilityReason(StrEnum):
    POLICY_UNAVAILABLE = "policy_unavailable"
    POLICY_INVALID = "policy_invalid"
    POLICY_DENIED = "policy_denied"
    PROVIDER_DENIED = "provider_denied"
    DATA_POLICY_DENIED = "data_policy_denied"
    CAPABILITY_MISSING = "capability_missing"
    QUALITY_INELIGIBLE = "quality_ineligible"
    BUDGET_INELIGIBLE = "budget_ineligible"
    DEADLINE_INELIGIBLE = "deadline_ineligible"


@dataclass(frozen=True)
class CandidateEligibility:
    candidate: ModelCandidate
    reasons: tuple[EligibilityReason, ...] = ()

    @property
    def eligible(self) -> bool:
        return not self.reasons
