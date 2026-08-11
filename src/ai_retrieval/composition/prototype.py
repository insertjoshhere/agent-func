"""Minimal vendor-neutral end-to-end prototype composition."""

import asyncio
from base64 import b64decode, b64encode
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json

from ai_retrieval.domain.budget import Usage

from ai_retrieval.admission.router import AdmissionRouter
from ai_retrieval.bulk import (
    BulkCoordinator, BulkWorkExecutor, CheckpointStage, EffectRecoveryCoordinator,
    InMemoryDeadLetterQueue, InMemoryDurableWorkRepository,
    InMemoryNotificationBroker, InMemoryObjectResultStore, JobTerminalCause,
    ObjectReference, TerminalWorkItemState, TransactionBoundary, WorkItemExecutionResult,
    WorkSubmission,
)
from ai_retrieval.bulk.worker import BulkStageAction, BulkStageResult, BulkWorker
from ai_retrieval.control_plane.budget import InMemoryBudgetController
from ai_retrieval.control_plane.configuration import (
    AdapterEvidenceRegistry, ConfigurationRegistry, ConfigurationValidator,
    CredentialAuthorizationRegistry, VersionRegistry,
)
from ai_retrieval.control_plane.context import DefaultExecutionContextFactory
from ai_retrieval.domain.admission import AdmissionEnvelope
from ai_retrieval.domain.budget import BudgetLimit
from ai_retrieval.domain.configuration import AdapterContractEvidence, CredentialAuthorizationEvidence
from ai_retrieval.domain.execution import ExecutionContext
from ai_retrieval.domain.immutable import FrozenMapping, freeze
from ai_retrieval.domain.model_routing import ModelCandidate, PolicyDecision, RoutingPolicy
from ai_retrieval.domain.outcomes import AdmissionDecision
from ai_retrieval.domain.work import (
    BatchJob, BulkTaskWork, InteractiveRequest, InteractiveTaskWork, ModelWork, QueryWork,
)
from ai_retrieval.interactive import (
    BoundBudgetAdapter, CandidateCatalog, CurrentProtectedModelExecutor,
    InteractiveCoordinator, InteractiveTaskProcessor, RedactedInteractiveTelemetry,
    RoutedModelPlanner, TaskAwareInteractiveCoordinator,
)
from ai_retrieval.model_routing import (
    CapacityLimit, CapacityScope, CapacityScopeKind, HierarchicalCapacity,
    ModelRouter,
)
from ai_retrieval.observability import ObservabilityService
from ai_retrieval.prototype import (
    AllowReadPolicy, FakeEffectAdapter, FakeModelProvider,
    FakeReadRelationalAdapter, FakeServiceAuthenticator, RecordingTelemetry,
)
from ai_retrieval.relational import (
    DataAccessLayer, NormalizedType, OperationKind, OperationNode, OrderTerm,
    QueryPlan, QueryPlanReference, QueryPlanRegistry, VendorNeutralContract,
)
from ai_retrieval.security import (
    ProtectedModelInvoker, SecurityDecision, SecurityPolicy, SecurityPolicyRegistry,
    SecurityRejection,
)
from ai_retrieval.tasks import (
    DeterministicTaskPacker, PackedRequest, PackingLimits, PreparedTask, RowInput,
    StructuredOutputParser, SummaryLengthLimits, TaskDefinition, TaskFailure,
    TaskFunction, TaskInvocation, TaskOutputFailure, TaskOutputValidator, TaskRuntime,
    build_seeded_task_definition_registry, canonical_definition_bytes,
    canonical_payload_bytes,
)
from ai_retrieval.validation import (
    DeterministicValidator, FieldRule, FieldType, ValidationRules,
    ValidationRulesRegistry,
)
from ai_retrieval.write_back import (
    ApprovalRecord, BatchWriteBackExecutor, InMemoryWriteBackAuthorization,
    WriteBackCommand, WriteBackExecutionStatus, WriteBackPolicy, WriteBackScope,
)


class _AvailableBudget:
    def is_available(self, operation, candidate) -> bool:
        return True


@dataclass(frozen=True)
class BulkDispatchResult:
    job_id: str
    item_count: int
    submitted_item_ids: tuple[str, ...]


@dataclass(frozen=True)
class BulkRunResult:
    job_id: str
    classification: str
    states: tuple[tuple[str, str], ...]
    attempts: tuple[tuple[str, int], ...]
    write_back_statuses: tuple[tuple[str, str], ...]
    mutation_count: int
    telemetry_count: int


class _BudgetBindingDispatcher:
    def __init__(self, coordinator, controller, scope: str) -> None:
        self._coordinator, self._controller, self._scope = coordinator, controller, scope

    def dispatch(self, request, context):
        self._controller.bind(context, (self._scope,))
        return self._coordinator.dispatch(request, context)


class _BulkDispatcher:
    def __init__(self, coordinator: BulkCoordinator, controller=None, scope: str = "prototype") -> None:
        self._coordinator = coordinator
        self._controller = controller
        self._scope = scope

    def dispatch(self, job: BatchJob, context: ExecutionContext) -> BulkDispatchResult:
        if self._controller is not None:
            self._controller.bind(context, (self._scope,))
        for item_id in job.item_ids:
            if job.task is None:
                payload = json.dumps({"id": item_id, "label": "ok"}, sort_keys=True).encode()
            else:
                source = job.task.source_rows.get(item_id)
                if not isinstance(source, FrozenMapping):
                    source = freeze({"text": item_id})
                    assert isinstance(source, FrozenMapping)
                payload = canonical_payload_bytes({
                    "kind": "task_invocation",
                    "function": job.task.function,
                    "parameters": job.task.parameters,
                    "rows": ({"id": item_id, "source": source},),
                    "requested_definition_version": job.task.requested_definition_version,
                })
            self._coordinator.submit(WorkSubmission(job.job_id, item_id, f"{job.job_id}:{item_id}", payload), context)
        self._coordinator.relay_pending()
        return BulkDispatchResult(job.job_id, len(job.item_ids), job.item_ids)


class _ObjectPayloadStore:
    """Task pack payload store backed by the prototype's content-addressed object store."""

    def __init__(self, objects: InMemoryObjectResultStore) -> None:
        self._objects = objects

    def put(self, payload: bytes, content_hash: str) -> str:
        reference = self._objects.put_payload(payload)
        if reference.content_hash != content_hash:
            raise ValueError("payload store content hash mismatch")
        return reference.reference


class _PrototypeTokenEstimator:
    def estimate_input(self, payload: bytes) -> int:
        return max((len(payload) + 3) // 4, 1)

    def estimate_output(self, definition: TaskDefinition, row_count: int) -> int:
        del definition
        return max(row_count * 8, 1)


class _BulkTaskFailure(RuntimeError):
    def __init__(self, state: TerminalWorkItemState, code: str, details: str) -> None:
        self.state, self.code, self.details = state, code, details
        super().__init__(details)


class _BulkPipeline:
    def __init__(self, app: "PrototypeApplication", write_back: BatchWriteBackExecutor) -> None:
        self._app, self._write_back = app, write_back
        self.statuses: dict[str, str] = {}

    def process_stage(self, claim, context):
        try:
            return self._process_stage(claim, context)
        except _BulkTaskFailure as failure:
            terminal = self._terminal(claim, failure.state, failure.code, failure.details)
            return BulkStageResult(BulkStageAction.TERMINAL, terminal=terminal, outcome=failure.code)
        except SecurityRejection as error:
            terminal = self._terminal(
                claim, TerminalWorkItemState.POLICY_REJECTED,
                error.failure.code.value, error.failure.message,
            )
            return BulkStageResult(BulkStageAction.TERMINAL, terminal=terminal, outcome="policy_rejected")
        except Exception as error:
            terminal = self._terminal(
                claim, TerminalWorkItemState.RETRY_EXHAUSTED,
                type(error).__name__, str(error),
            )
            return BulkStageResult(BulkStageAction.TERMINAL, terminal=terminal, outcome="retry_exhausted")

    def _process_stage(self, claim, context):
        state = self._app.repository.resume_state(claim.item.job_id, claim.item.idempotency_key)
        stage = state.next_stage
        if not self._is_task(claim.item.payload_reference):
            return self._process_legacy(stage, claim, context)
        if stage is CheckpointStage.ACCEPTED:
            prepared = self._prepare_task(claim, context)
            reference = self._app.objects.put_result(self._prepared_bytes(prepared))
            return self._checkpoint(claim, stage, "task_prepared", reference)
        if stage is CheckpointStage.MODEL_COMPLETED:
            prepared = self._load_prepared(claim.item.result_reference)
            responses = self._invoke_packs(prepared, context)
            reference = self._app.objects.put_result(canonical_payload_bytes(responses))
            return self._checkpoint(claim, stage, "model_succeeded", reference)
        if stage is CheckpointStage.RESULT_STORED:
            prepared = self._prepared_from_accepted_checkpoint(claim)
            responses = self._load_json(claim.item.result_reference)
            parsed = self._parse_responses(prepared, responses)
            reference = self._app.objects.put_result(canonical_payload_bytes(parsed))
            return self._checkpoint(claim, stage, "parsed", reference)
        if stage is CheckpointStage.VALIDATED:
            prepared = self._prepared_from_accepted_checkpoint(claim)
            parsed = self._load_json(claim.item.result_reference)
            validation = self._validate_responses(prepared, parsed, context)
            if not validation.outcome.accepted:
                details = ",".join(validation.outcome.reason_codes or validation.outcome.failed_rule_ids)
                raise _BulkTaskFailure(
                    TerminalWorkItemState.VALIDATION_FAILED, "validation_failed", details,
                )
            accepted = tuple(dict(record) for record in validation.accepted_records)
            reference = self._app.objects.put_result(canonical_payload_bytes(accepted))
            return self._checkpoint(claim, stage, "accepted", reference)
        if stage is CheckpointStage.COMPLETED:
            prepared = self._prepared_from_accepted_checkpoint(claim)
            accepted = self._load_json(claim.item.result_reference)
            validation = self._validate_responses(prepared, accepted, context, accepted_flat=True)
            return self._write_accepted(claim, context, validation, prepared.invocation.function)
        raise RuntimeError(f"unsupported resume stage: {stage}")

    def _process_legacy(self, stage, claim, context):
        if stage is CheckpointStage.ACCEPTED:
            checkpoint = self._app.repository.checkpoint(
                claim.item.job_id, claim.item.idempotency_key, claim.item.lease_owner,
                CheckpointStage.ACCEPTED, "accepted", self._app.clock(),
            )
            return BulkStageResult(BulkStageAction.CHECKPOINTED, checkpoint.completed_stage, outcome=checkpoint.outcome)
        if stage is CheckpointStage.MODEL_COMPLETED:
            checkpoint = self._app.repository.checkpoint(
                claim.item.job_id, claim.item.idempotency_key, claim.item.lease_owner,
                CheckpointStage.MODEL_COMPLETED, "model_succeeded", self._app.clock(),
                claim.item.payload_reference,
            )
            return BulkStageResult(BulkStageAction.CHECKPOINTED, checkpoint.completed_stage, outcome=checkpoint.outcome)
        if stage is CheckpointStage.RESULT_STORED:
            checkpoint = self._app.repository.checkpoint(
                claim.item.job_id, claim.item.idempotency_key, claim.item.lease_owner,
                CheckpointStage.RESULT_STORED, "result_stored", self._app.clock(),
                claim.item.result_reference,
            )
            return BulkStageResult(BulkStageAction.CHECKPOINTED, checkpoint.completed_stage, outcome=checkpoint.outcome)
        if stage is CheckpointStage.VALIDATED:
            output = json.loads(self._app.objects.get(claim.item.result_reference))
            validation = self._app.validator.validate((claim.item.item_id,), (output,), context)
            if not validation.outcome.accepted:
                terminal = self._terminal(
                    claim, TerminalWorkItemState.VALIDATION_FAILED,
                    "validation_failed", "invalid output",
                )
                return BulkStageResult(BulkStageAction.TERMINAL, terminal=terminal, outcome="validation_failed")
            checkpoint = self._app.repository.checkpoint(
                claim.item.job_id, claim.item.idempotency_key, claim.item.lease_owner,
                CheckpointStage.VALIDATED, "accepted", self._app.clock(), claim.item.result_reference,
            )
            return BulkStageResult(BulkStageAction.CHECKPOINTED, checkpoint.completed_stage, outcome=checkpoint.outcome)
        if stage is CheckpointStage.COMPLETED:
            validation = self._app.validator.validate(
                (claim.item.item_id,),
                (json.loads(self._app.objects.get(claim.item.result_reference)),), context,
            )
            return self._write_accepted(claim, context, validation, None)
        raise RuntimeError(f"unsupported resume stage: {stage}")

    def _prepare_task(self, claim, context) -> PreparedTask:
        data = self._load_json(claim.item.payload_reference)
        try:
            function = TaskFunction(data["function"])
            parameters = freeze(data["parameters"])
            rows = tuple(
                RowInput(row["id"], freeze(row["source"])) for row in data["rows"]
            )
            assert isinstance(parameters, FrozenMapping)
            invocation = TaskInvocation(
                function, parameters, rows, data.get("requested_definition_version"),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise _BulkTaskFailure(
                TerminalWorkItemState.VALIDATION_FAILED,
                "invalid_task_invocation", type(error).__name__,
            ) from error
        prepared = self._app.task_runtime.prepare(invocation, context)
        if isinstance(prepared, TaskFailure):
            state = (
                TerminalWorkItemState.POLICY_REJECTED
                if prepared.stage.value == "resolution"
                else TerminalWorkItemState.VALIDATION_FAILED
            )
            raise _BulkTaskFailure(state, prepared.code.value, self._failure_details(prepared))
        return prepared

    def _invoke_packs(self, prepared: PreparedTask, context) -> tuple[dict[str, object], ...]:
        responses: list[dict[str, object]] = []
        for pack, work in zip(prepared.packs, prepared.model_work, strict=True):
            plan = None
            reservation_id = None
            actual = None
            try:
                operation_id = f"task:{pack.function.value}:{pack.definition_version}:pack:{pack.pack_index}"
                plan = self._app.model_planner.plan(operation_id, work, context, 0)
                decision = self._app.budget_adapter.reserve(context, plan.estimate)
                if not decision.accepted:
                    scopes = ",".join(decision.exhaustion.exhausted_scopes)
                    raise _BulkTaskFailure(
                        TerminalWorkItemState.BUDGET_EXHAUSTED, "budget_exhausted", scopes,
                    )
                reservation_id = decision.reservation.reservation_id
                invoked = asyncio.run(self._app.model_executor.invoke(plan, None, context))
                actual = invoked.actual_usage
                output = invoked.output
                if isinstance(output, bytes):
                    responses.append({
                        "pack_index": pack.pack_index,
                        "response_encoding": "base64-bytes",
                        "output": b64encode(output).decode("ascii"),
                    })
                else:
                    responses.append({
                        "pack_index": pack.pack_index,
                        "response_encoding": "json",
                        "output": output,
                    })
            finally:
                if reservation_id is not None:
                    self._app.budget_adapter.reconcile(reservation_id, actual or plan.estimate)
                if plan is not None:
                    self._app.model_planner.complete(plan)
        return tuple(responses)

    def _parse_responses(self, prepared, responses):
        by_index = {}
        for item in responses:
            encoding = item.get("response_encoding", "json")
            output = item["output"]
            if encoding == "base64-bytes":
                output = b64decode(output, validate=True)
            elif encoding != "json":
                raise _BulkTaskFailure(
                    TerminalWorkItemState.VALIDATION_FAILED,
                    "unsupported_representation", "unknown checkpointed response encoding",
                )
            by_index[item["pack_index"]] = output
        parsed: list[dict[str, object]] = []
        for pack in prepared.packs:
            result = self._app.task_parser.parse(by_index[pack.pack_index], prepared.definition)
            if isinstance(result, TaskOutputFailure):
                raise _BulkTaskFailure(
                    TerminalWorkItemState.VALIDATION_FAILED,
                    result.code.value, self._failure_details(result),
                )
            parsed.append({"pack_index": pack.pack_index, "records": result})
        return tuple(parsed)

    def _validate_responses(self, prepared, parsed, context, *, accepted_flat=False):
        if accepted_flat:
            records = tuple(parsed)
            outputs = []
            start = 0
            for pack in prepared.packs:
                stop = start + len(pack.input_ids)
                outputs.append((pack, records[start:stop]))
                start = stop
        else:
            by_index = {item["pack_index"]: item["records"] for item in parsed}
            outputs = [(pack, by_index[pack.pack_index]) for pack in prepared.packs]
        return self._app.task_validator.validate_packs(prepared, outputs, context)

    def _write_accepted(self, claim, context, validation, function):
        field = "summary" if function is TaskFunction.SUMMARIZE else "label"
        value = validation.accepted_records[0][field]
        policy = "write-policy-summary" if field == "summary" else "write-policy-1"
        approval = "approval-summary" if field == "summary" else "approval-1"
        command = WriteBackCommand(
            claim, validation, policy, approval, "prototype",
            "results", f"id={claim.item.item_id}", frozenset({field}),
            OperationKind.UPDATE, ((field, value),), TransactionBoundary.SHARED,
        )
        outcome = self._write_back.execute(command, context)
        self.statuses[claim.item.item_id] = outcome.status.value
        if outcome.status in {WriteBackExecutionStatus.COMMITTED, WriteBackExecutionStatus.RECOVERED}:
            return BulkStageResult(BulkStageAction.CHECKPOINTED, CheckpointStage.COMPLETED, outcome=outcome.status.value)
        if outcome.status is WriteBackExecutionStatus.PERSISTED:
            terminal = self._app.bulk_executor.execute_claim(claim, _ResultHandler(
                WorkItemExecutionResult(
                    TerminalWorkItemState.SUCCEEDED,
                    result=self._app.objects.get(outcome.output_reference),
                )
            ))
            return BulkStageResult(BulkStageAction.TERMINAL, terminal=terminal, outcome=outcome.status.value)
        terminal = self._terminal(
            claim, TerminalWorkItemState.WRITE_BACK_FAILED,
            outcome.status.value, ",".join(outcome.reason_codes),
        )
        return BulkStageResult(BulkStageAction.TERMINAL, terminal=terminal, outcome=outcome.status.value)

    def _prepared_from_accepted_checkpoint(self, claim):
        checkpoints = self._app.repository.checkpoints(
            claim.item.job_id, claim.item.idempotency_key,
        )
        accepted = next(
            checkpoint for checkpoint in checkpoints
            if checkpoint.completed_stage is CheckpointStage.ACCEPTED
        )
        return self._load_prepared(accepted.result_reference)

    def _load_prepared(self, reference) -> PreparedTask:
        data = self._load_json(reference)
        function = TaskFunction(data["function"])
        definition = self._app.task_registry.resolve(function, data["definition_version"])
        if definition is None:
            raise _BulkTaskFailure(
                TerminalWorkItemState.POLICY_REJECTED,
                "task_definition_unavailable", "checkpointed definition unavailable",
            )
        if sha256(canonical_definition_bytes(definition)).hexdigest() != data["definition_hash"]:
            raise _BulkTaskFailure(
                TerminalWorkItemState.POLICY_REJECTED,
                "task_definition_incompatible", "checkpointed definition mismatch",
            )
        parameters = freeze(data["parameters"])
        assert isinstance(parameters, FrozenMapping)
        rows = tuple(RowInput(row["id"], freeze(row["source"])) for row in data["rows"])
        invocation = TaskInvocation(function, parameters, rows, data.get("requested_definition_version"))
        packs, works = [], []
        for item in data["packs"]:
            payload_reference = item["payload_reference"]
            payload_ref = self._app.objects.resolve(payload_reference)
            payload = self._app.objects.get(payload_ref)
            if sha256(payload).hexdigest() != item["payload_hash"]:
                raise ValueError("checkpointed task payload hash mismatch")
            pack = PackedRequest(
                function, definition.version, item["pack_index"], tuple(item["input_ids"]),
                payload, item["estimated_input_tokens"], item["estimated_output_tokens"],
            )
            packs.append(pack)
            works.append(ModelWork(
                function.value, payload_reference, pack.input_ids,
                definition.required_capabilities, definition.version,
                pack.estimated_input_tokens, pack.estimated_output_tokens,
            ))
        return PreparedTask(definition, invocation, tuple(packs), tuple(works))

    def _prepared_bytes(self, prepared):
        return canonical_payload_bytes({
            "kind": "prepared_task",
            "function": prepared.definition.function.value,
            "definition_version": prepared.definition.version,
            "definition_hash": sha256(canonical_definition_bytes(prepared.definition)).hexdigest(),
            "parameters": prepared.invocation.parameters,
            "rows": tuple(
                {"id": row.identifier, "source": row.source_fields}
                for row in prepared.invocation.rows
            ),
            "requested_definition_version": prepared.invocation.requested_definition_version,
            "packs": tuple({
                "pack_index": pack.pack_index,
                "input_ids": pack.input_ids,
                "payload_reference": work.payload_reference,
                "payload_hash": sha256(pack.payload).hexdigest(),
                "estimated_input_tokens": pack.estimated_input_tokens,
                "estimated_output_tokens": pack.estimated_output_tokens,
            } for pack, work in zip(prepared.packs, prepared.model_work, strict=True)),
        })

    def _checkpoint(self, claim, stage, outcome, reference):
        checkpoint = self._app.repository.checkpoint(
            claim.item.job_id, claim.item.idempotency_key, claim.item.lease_owner,
            stage, outcome, self._app.clock(), reference,
        )
        return BulkStageResult(
            BulkStageAction.CHECKPOINTED, checkpoint.completed_stage, outcome=checkpoint.outcome,
        )

    def _terminal(self, claim, state, code, details):
        return self._app.bulk_executor.execute_claim(claim, _ResultHandler(
            WorkItemExecutionResult(state, code, details)
        ))

    def _is_task(self, reference: ObjectReference) -> bool:
        try:
            return self._load_json(reference).get("kind") == "task_invocation"
        except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
            return False

    def _load_json(self, reference):
        return json.loads(self._app.objects.get(reference).decode("utf-8"))

    @staticmethod
    def _failure_details(failure):
        return json.dumps({
            "stage": failure.stage.value,
            "rules": failure.failed_rule_ids,
            "details": dict(failure.details),
        }, sort_keys=True, default=str)


class _ResultHandler:
    def __init__(self, result) -> None:
        self._result = result

    def execute(self, item):
        return self._result


class PrototypeApplication:
    def __init__(self, *, write_back_enabled: bool = False, model_valid: bool = True) -> None:
        self._now = datetime.now(timezone.utc)
        self.clock = lambda: self._now
        self.telemetry = RecordingTelemetry()
        self.read_adapter = FakeReadRelationalAdapter()
        self.model_provider = FakeModelProvider(not model_valid)
        self.effect_adapter = FakeEffectAdapter(self.clock)
        self.objects = InMemoryObjectResultStore()
        self.repository = InMemoryDurableWorkRepository()
        self.validation_registry = ValidationRulesRegistry((ValidationRules("rules-1", "id", (
            FieldRule("id.string", "id", FieldType.STRING),
            FieldRule("label.string", "label", FieldType.STRING, required=False),
            FieldRule("summary.string", "summary", FieldType.STRING, required=False),
        )),))
        self.task_registry = build_seeded_task_definition_registry(
            self.validation_registry, "rules-1",
        )
        definition_versions = freeze({
            definition.function.value: definition.version
            for definition in self.task_registry.registered_definitions
        })
        assert isinstance(definition_versions, FrozenMapping)
        self.task_definition_versions = definition_versions
        self.configuration = self._configuration(write_back_enabled)
        self.reference = self.configuration.active_reference("prototype")
        ids = iter(f"prototype-{index}" for index in range(1, 10_000))
        self.broker = InMemoryNotificationBroker(lambda: next(ids))
        self.bulk_coordinator = BulkCoordinator(self.repository, self.objects, self.broker, self.clock, lambda: next(ids))
        self.validator = DeterministicValidator(self.validation_registry, self.telemetry)
        self.task_validator = TaskOutputValidator(self.validator)
        self.task_parser = StructuredOutputParser()
        self.token_estimator = _PrototypeTokenEstimator()
        self.payload_store = _ObjectPayloadStore(self.objects)
        self.task_runtime = TaskRuntime(
            self.task_registry,
            DeterministicTaskPacker(self.token_estimator, self.payload_store),
        )
        self.budget = InMemoryBudgetController((BudgetLimit("prototype", 100, 10_000),))
        interactive = self._interactive()
        self.bulk_executor = BulkWorkExecutor(
            self.repository, self.objects, InMemoryDeadLetterQueue(), self.telemetry, self.clock, self.clock(),
        )
        self.write_back = self._write_back_executor()
        self.pipeline = _BulkPipeline(self, self.write_back)
        self.worker = BulkWorker(self.repository, self.pipeline, self.clock)
        self.router = AdmissionRouter(
            self.configuration,
            DefaultExecutionContextFactory(clock=self.clock, identifier=lambda: next(ids)),
            _BudgetBindingDispatcher(interactive, self.budget, "prototype"),
            _BulkDispatcher(self.bulk_coordinator, self.budget, "prototype"),
        )

    def admit_interactive(self, request_id: str, plan_id: str, deadline_seconds: float, *, with_model: bool = True):
        model_work = (ModelWork("extract", "inline", ("customer-1",), frozenset({"extract"})),) if with_model else ()
        return self.router.admit(AdmissionEnvelope(
            self.reference,
            interactive=InteractiveRequest(request_id, QueryWork(plan_id), model_work),
            interactive_deadline_seconds=deadline_seconds,
        ))

    def admit_interactive_task(
        self, request_id: str, plan_id: str, deadline_seconds: float,
        function: TaskFunction | str, parameters: FrozenMapping,
        *, requested_definition_version: str | None = None,
    ):
        return self.router.admit(AdmissionEnvelope(
            self.reference,
            interactive=InteractiveRequest(
                request_id, QueryWork(plan_id), (),
                InteractiveTaskWork(str(function), parameters, requested_definition_version),
            ),
            interactive_deadline_seconds=deadline_seconds,
        ))

    def admit_bulk(self, job_id: str, item_ids: tuple[str, ...]):
        return self.router.admit(AdmissionEnvelope(self.reference, bulk=BatchJob(job_id, item_ids)))

    def admit_bulk_task(
        self, job_id: str, item_ids: tuple[str, ...],
        function: TaskFunction | str, parameters: FrozenMapping,
        *, source_rows: FrozenMapping | None = None,
        requested_definition_version: str | None = None,
    ):
        rows = source_rows if source_rows is not None else FrozenMapping(())
        task = BulkTaskWork(
            str(function), parameters, rows, requested_definition_version,
        )
        return self.router.admit(AdmissionEnvelope(
            self.reference, bulk=BatchJob(job_id, item_ids, task=task),
        ))

    def run_bulk(self, decision: AdmissionDecision, *, simulate_interrupt: bool = False) -> BulkRunResult:
        if decision.context is None or not isinstance(decision.outcome, BulkDispatchResult):
            raise ValueError("an accepted bulk admission decision is required")
        job_id = decision.outcome.job_id
        for item_id in decision.outcome.submitted_item_ids:
            key = f"{job_id}:{item_id}"
            if simulate_interrupt:
                self.worker.resume(job_id, key, "worker-before-crash", decision.context, max_stages=1)
            self.worker.resume(job_id, key, "worker-resumed", decision.context)
        terminals = self.repository.terminal_items(job_id)
        if len(terminals) == len(decision.outcome.submitted_item_ids):
            cause = (
                JobTerminalCause.BUDGET_EXHAUSTED
                if any(item.state is TerminalWorkItemState.BUDGET_EXHAUSTED for item in terminals)
                else JobTerminalCause.COMPLETED
            )
            report = self.bulk_executor.terminal_report(job_id, cause)
            classification = report.classification.value
        else:
            classification = "succeeded" if all(self.repository.resume_state(job_id, f"{job_id}:{item}").next_stage is None for item in decision.outcome.submitted_item_ids) else "in-progress"
        records = self.repository.items(job_id)
        terminal_by_id = {item.item_id: item.state.value for item in terminals}
        states = tuple((
            item.item_id,
            terminal_by_id.get(
                item.item_id,
                "succeeded" if self.repository.resume_state(job_id, item.idempotency_key).next_stage is None else item.state.value,
            ),
        ) for item in records)
        attempts = tuple((item.item_id, item.attempt_count) for item in records)
        return BulkRunResult(
            job_id, classification, states, attempts,
            tuple(sorted(self.pipeline.statuses.items())), self.effect_adapter.mutation_count,
            len(self.telemetry.events),
        )

    def _interactive(self):
        plan = QueryPlan(
            QueryPlanReference("customers", "1"), OperationNode(OperationKind.SELECT, dataset="customers"),
            (), frozenset({"typed_parameters", "deterministic_order"}), (OrderTerm("id"),),
            False, 100, 100_000, frozenset({"customers"}),
        )
        dal = DataAccessLayer(
            QueryPlanRegistry((plan,)), VendorNeutralContract(
                "1", self.read_adapter.capabilities(), frozenset(NormalizedType),
            ), self.read_adapter, AllowReadPolicy(), self.telemetry, self.clock,
        )
        candidates = (
            ModelCandidate(
                "cheap", "fake-provider",
                frozenset({"extract", "ai_classify", "ai_summarize"}),
                frozenset(), 1, 10, 10, 0.9,
            ),
            ModelCandidate(
                "fallback", "fake-provider",
                frozenset({"extract", "ai_classify", "ai_summarize"}),
                frozenset(), 2, 20, 10, 1.0,
            ),
        )
        routing_policy = RoutingPolicy("routing-1", PolicyDecision.ALLOW, True, True, frozenset({"fake-provider"}), frozenset())
        router = ModelRouter(
            _AvailableBudget(), HierarchicalCapacity({CapacityScope(CapacityScopeKind.SYSTEM, "system"): CapacityLimit(concurrency=4)}),
        )
        policies = SecurityPolicyRegistry((SecurityPolicy(
            "security-1", True, True, SecurityDecision.ALLOW,
            frozenset({"fake-provider"}), frozenset(), frozenset({"secret"}),
            transport_encryption="tls", persistence_encryption="memory-encryption",
        ),))
        observability = ObservabilityService(self.telemetry)
        invoker = ProtectedModelInvoker(policies, FakeServiceAuthenticator(), self.model_provider, observability, self.clock)
        self.model_planner = RoutedModelPlanner(
            router,
            CandidateCatalog(lambda work, context: candidates, lambda context: routing_policy),
            self.clock,
        )
        self.model_executor = CurrentProtectedModelExecutor(
            invoker, self._model_payload,
        )
        self.budget_adapter = BoundBudgetAdapter(self.budget)
        task_processor = InteractiveTaskProcessor(
            self.task_runtime, self.task_parser, self.task_validator,
        )
        return TaskAwareInteractiveCoordinator(
            dal, self.model_planner, self.model_executor,
            self.budget_adapter, self.validator,
            RedactedInteractiveTelemetry(observability, policies, self.clock),
            task_processor, response_reserve_ms=1,
        )

    def _model_payload(self, plan, source):
        work = plan.operation.work
        if work.task_definition_version is not None:
            reference = self.objects.resolve(work.payload_reference)
            return json.loads(self.objects.get(reference).decode("utf-8"))
        return {"rows": source.rows}

    def _write_back_executor(self):
        classify_scope = WriteBackScope(
            "bulk-job", frozenset({"item-1"}), "prototype", "results", "id=item-1",
            frozenset({"label"}), OperationKind.UPDATE, "rules-1", self.reference.version,
            self.clock() - timedelta(minutes=1), self.clock() + timedelta(hours=1),
        )
        summary_scope = WriteBackScope(
            "bulk-job", frozenset({"item-1"}), "prototype", "results", "id=item-1",
            frozenset({"summary"}), OperationKind.UPDATE, "rules-1", self.reference.version,
            self.clock() - timedelta(minutes=1), self.clock() + timedelta(hours=1),
        )
        authorization = InMemoryWriteBackAuthorization(
            (
                WriteBackPolicy("write-policy-1", classify_scope),
                WriteBackPolicy("write-policy-summary", summary_scope),
            ),
            (
                ApprovalRecord("approval-1", "prototype-owner", "write-policy-1", classify_scope),
                ApprovalRecord("approval-summary", "prototype-owner", "write-policy-summary", summary_scope),
            ),
        )
        recovery = EffectRecoveryCoordinator(self.repository, self.repository, self.effect_adapter, self.clock)
        return BatchWriteBackExecutor(self.objects, authorization, recovery, self.clock, self.telemetry)

    def _configuration(self, write_back_enabled):
        evidence_registry = AdapterEvidenceRegistry()
        evidence_registry.activate(AdapterContractEvidence(
            "prototype-memory", "1", "contract-1", ("select",), ("string",), ("null",),
            ("explicit",), ("typed",), ("commit",), ("rollback",), (("select", True),), self.clock(),
        ))
        validator = ConfigurationValidator(
            VersionRegistry(("security-1",)), VersionRegistry(("rules-1",)),
            VersionRegistry(("model-1",)), evidence_registry,
            CredentialAuthorizationRegistry((CredentialAuthorizationEvidence("prototype-reader", frozenset({"select"})),)),
        )
        registry = ConfigurationRegistry(validator, timedelta(days=1), self.clock)
        registry.activate("prototype", {
            "routing": {"interactive": True, "bulk": True},
            "database": {"adapter_id": "prototype-memory", "adapter_version": "1", "read_only_credential_id": "prototype-reader", "query_plan_version": "1"},
            "model": {"versions": ["model-1"]},
            "budget": {"cost_limit": 100, "token_limit": 10_000},
            "deadline": {"duration_seconds": 2}, "cancellation": {"timeout_seconds": 0},
            "concurrency": {"limit": 4}, "retry": {"max_attempts": 2},
            "security": {"policy_version": "security-1"}, "validation": {"rules_version": "rules-1"},
            "telemetry": {"enabled": True}, "write_back": {"enabled": write_back_enabled},
            "task_definitions": self.task_definition_versions,
            "interactive": {
                "tenant_id": "prototype", "estimated_model_tokens": 4,
                "estimated_output_tokens": 1, "model_service_identity": "prototype-model",
                "task_input": {
                    "identifier_column": "id",
                    "source_fields": {"text": "name"},
                },
            },
        })
        return registry


def build_prototype(*, write_back_enabled: bool = False, model_valid: bool = True) -> PrototypeApplication:
    return PrototypeApplication(write_back_enabled=write_back_enabled, model_valid=model_valid)
