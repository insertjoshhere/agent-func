"""Local replaceable adapters used only by the executable prototype."""

from ai_retrieval.bulk import (
    EffectEvidence, EffectEvidenceStatus, EffectRecord, TransactionBoundary,
)
from ai_retrieval.domain.budget import Usage
from ai_retrieval.domain.immutable import FrozenMapping
from ai_retrieval.interactive import ModelInvocationResult
from ai_retrieval.relational import (
    DatabaseAccessDecision, NormalizedType, RawColumn, RawRelationalResult,
)
from ai_retrieval.security import AuthenticationResult


class RecordingTelemetry:
    """In-memory sink implementing the prototype's telemetry/audit ports."""

    def __init__(self) -> None:
        self.events: list[object] = []

    async def emit(self, event) -> None:
        self.events.append(event)

    def record(self, metadata) -> None:
        self.events.append(metadata)

    def job_state_changed(self, telemetry) -> None:
        self.events.append(telemetry)

    def append(self, event) -> None:
        self.events.append(event)


class FakeReadRelationalAdapter:
    adapter_id = "prototype-memory"
    adapter_version = "1"

    def __init__(self) -> None:
        self.read_count = 0
        self.cancelled: list[str] = []

    def capabilities(self) -> frozenset[str]:
        return frozenset({"operation:select", "typed_parameters", "deterministic_order"})

    async def execute_read(self, plan, parameters, access, context):
        self.read_count += 1
        return RawRelationalResult(
            (RawColumn("id", NormalizedType.STRING, False), RawColumn("name", NormalizedType.STRING, False)),
            (("customer-1", "Ada"),),
        )

    async def cancel(self, cancellation_token: str) -> None:
        self.cancelled.append(cancellation_token)


class AllowReadPolicy:
    async def authorize_read(self, credential_id, plan, context):
        return DatabaseAccessDecision(
            True, credential_id, "prototype-reader", "security-1", "allow", "tls", "memory-encryption",
        )


class FakeServiceAuthenticator:
    async def authenticate(self, service_identity, provider_id, context):
        return AuthenticationResult(True, service_identity, "allow")


class FakeModelProvider:
    """A deterministic provider fake; output remains behind the secured gateway port."""

    def __init__(self, invalid_output: bool = False) -> None:
        self.invalid_output = invalid_output
        self.invalid_item_ids: set[str] = set()
        self.calls = 0

    async def invoke(self, request, context):
        self.calls += 1
        work = request.operation.work
        invalid = self.invalid_output or any(
            item_id in self.invalid_item_ids for item_id in work.input_ids
        )
        if invalid:
            return ModelInvocationResult(
                tuple({"id": item_id, "unexpected": "invalid"} for item_id in work.input_ids),
                Usage(1, 3, 1),
            )
        if work.task_type == "ai_classify":
            labels = request.payload.get("parameters", FrozenMapping(())).get("labels", ("ok",))
            label = labels[0] if labels else "ok"
            output = tuple({"id": item_id, "label": label} for item_id in work.input_ids)
        elif work.task_type == "ai_summarize":
            parameters = request.payload.get("parameters", FrozenMapping(()))
            max_words = parameters.get("max_words", 1)
            rows = request.payload.get("rows", ())
            source_by_id = {
                row.get("id"): row.get("source", FrozenMapping(())).get("text", "")
                for row in rows
            }
            output = tuple(
                {"id": item_id, "summary": " ".join(str(source_by_id.get(item_id, "")).split()[:max_words])}
                for item_id in work.input_ids
            )
        else:
            output = tuple({"id": item_id, "label": "ok"} for item_id in work.input_ids)
        return ModelInvocationResult(output, Usage(1, 3, 1))


class FakeEffectAdapter:
    """Shared-transaction fake proving effect recovery without vendor coupling."""

    boundary = TransactionBoundary.SHARED
    adapter_version = "prototype-memory-1"

    def __init__(self, clock) -> None:
        self._clock = clock
        self.mutation_count = 0

    def execute_shared(self, request, context):
        self.mutation_count += 1
        item = request.claim.item
        return EffectEvidence(EffectEvidenceStatus.COMMITTED, EffectRecord(
            item.job_id, item.item_id, item.idempotency_key,
            request.target_dataset, request.row_scope_digest, request.mutation_digest,
            1, "committed", self.adapter_version, self._clock(),
        ))

    def verify_effect(self, idempotency_key, mutation_digest, context):
        return EffectEvidence(EffectEvidenceStatus.ABSENT)

    def execute_non_shared(self, effect, mutation_digest, context):
        raise AssertionError("prototype uses the shared transaction boundary")
