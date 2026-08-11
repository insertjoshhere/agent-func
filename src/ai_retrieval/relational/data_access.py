"""Fail-closed vendor-neutral relational data access layer."""

from collections.abc import Callable, Mapping
from datetime import datetime, timezone

from ai_retrieval.domain.execution import ExecutionContext, ExecutionPath
from ai_retrieval.domain.failures import FailureCode, TypedFailure
from ai_retrieval.domain.immutable import FrozenMapping
from ai_retrieval.relational.models import (
    MUTATION_OPERATIONS,
    DatabaseAccessDecision,
    OperationClassification,
    ProtectionMetadata,
    QueryPlan,
    QueryPlanReference,
    SecurityAuditEvent,
    VendorNeutralContract,
)
from ai_retrieval.relational.normalization import NormalizationError, normalize_result
from ai_retrieval.relational.plans import QueryPlanRegistry, bind_parameters, classify_operation, operation_kinds
from ai_retrieval.relational.ports import DatabaseAccessPolicy, ReadRelationalAdapter, SecurityAuditSink


class DataAccessRejection(RuntimeError):
    def __init__(self, failure: TypedFailure) -> None:
        super().__init__(failure.message)
        self.failure = failure


class DataAccessLayer:
    """The only read submission path; it has no mutation-capable dependency."""

    def __init__(
        self,
        plans: QueryPlanRegistry,
        contract: VendorNeutralContract,
        adapter: ReadRelationalAdapter,
        access_policy: DatabaseAccessPolicy,
        audit_sink: SecurityAuditSink,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._plans = plans
        self._contract = contract
        self._adapter = adapter
        self._access_policy = access_policy
        self._audit_sink = audit_sink
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    async def execute_read(
        self,
        reference: QueryPlanReference,
        parameters: Mapping[str, object],
        context: ExecutionContext,
    ):
        plan = self._plans.resolve(reference)
        if plan is None:
            self._reject(FailureCode.QUERY_PLAN_UNAVAILABLE, f"query plan {reference.plan_id}@{reference.version} is not allowlisted")

        classification = classify_operation(plan.operation)
        if classification is OperationClassification.MUTATION:
            await self._security_reject(context, plan, FailureCode.MUTATION_BLOCKED, "mutation_operation")
        if classification is not OperationClassification.RETRIEVAL:
            self._reject(FailureCode.UNSUPPORTED_OPERATION, "query plan contains an unsupported operation")

        required = set(plan.required_capabilities)
        required.update(f"operation:{kind.value}" for kind in operation_kinds(plan.operation))
        missing_contract = required - self._contract.capabilities
        if missing_contract:
            self._reject(FailureCode.UNSUPPORTED_CAPABILITY, f"capabilities absent from contract: {', '.join(sorted(missing_contract))}")
        missing_adapter = required - self._adapter.capabilities()
        if missing_adapter:
            self._reject(FailureCode.UNSUPPORTED_CAPABILITY, f"adapter lacks capabilities: {', '.join(sorted(missing_adapter))}")
        if not plan.intrinsically_ordered and not plan.deterministic_order:
            self._reject(FailureCode.NONDETERMINISTIC_ORDER, "comparable results require deterministic ordering")

        configured_credential = self._credential_id(context)
        if context.path is not ExecutionPath.INTERACTIVE:
            self._reject(FailureCode.READ_ONLY_CREDENTIAL_REQUIRED, "read DAL accepts interactive execution context only")
        try:
            bound = bind_parameters(plan, parameters)
        except ValueError as error:
            self._reject(FailureCode.INVALID_QUERY_PARAMETERS, str(error))

        access = await self._access_policy.authorize_read(configured_credential, plan, context)
        if not access.allowed or access.credential_id != configured_credential:
            await self._security_reject(context, plan, FailureCode.DATABASE_ACCESS_DENIED, access.reason_code, access)
        if not access.service_identity.strip():
            await self._security_reject(context, plan, FailureCode.DATABASE_AUTHENTICATION_FAILED, "service_identity_missing", access)
        if not access.transport_encryption or not access.persistence_encryption:
            await self._security_reject(context, plan, FailureCode.DATABASE_PROTECTION_REQUIRED, "encryption_metadata_missing", access)

        raw = await self._adapter.execute_read(plan, bound, access, context)
        protection = ProtectionMetadata(access.policy_version, access.service_identity, access.transport_encryption, access.persistence_encryption)
        try:
            return normalize_result(raw, plan, protection, self._adapter.adapter_version)
        except NormalizationError as error:
            self._reject(FailureCode.RESULT_NORMALIZATION_FAILED, str(error))

    async def cancel(self, cancellation_token: str) -> None:
        """Forward cooperative cancellation without exposing mutation capability."""
        await self._adapter.cancel(cancellation_token)

    def _credential_id(self, context: ExecutionContext) -> str:
        database = context.configuration.content.get("database")
        credential = database.get("read_only_credential_id") if isinstance(database, FrozenMapping) else None
        if not isinstance(credential, str) or not credential.strip():
            self._reject(FailureCode.READ_ONLY_CREDENTIAL_REQUIRED, "bound read-only credential is unavailable")
        return credential

    async def _security_reject(
        self, context: ExecutionContext, plan: QueryPlan, code: FailureCode, reason: str,
        access: DatabaseAccessDecision | None = None,
    ) -> None:
        event = SecurityAuditEvent(
            correlation_id=str(context.correlation_id),
            configuration_version=context.configuration.reference.version,
            policy_version=access.policy_version if access else (context.configuration.security_policy_version or "unavailable"),
            service_identity=access.service_identity if access else "unavailable",
            decision="rejected", reason_code=reason, timestamp=self._clock(),
            operation_classification=classify_operation(plan.operation), plan_reference=plan.reference,
        )
        try:
            await self._audit_sink.emit(event)
        finally:
            self._reject(code, reason)

    @staticmethod
    def _reject(code: FailureCode, message: str):
        raise DataAccessRejection(TypedFailure(code, message))
