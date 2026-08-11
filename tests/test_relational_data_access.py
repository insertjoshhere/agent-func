import asyncio
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from ai_retrieval.domain.configuration import ConfigurationReference, ExecutionConfiguration
from ai_retrieval.domain.execution import CancellationContext, DeadlineContext, ExecutionContext, ExecutionPath
from ai_retrieval.domain.failures import FailureCode
from ai_retrieval.domain.identifiers import CorrelationId, ExecutionId
from ai_retrieval.domain.immutable import freeze
from ai_retrieval.relational import (
    DataAccessLayer, DataAccessRejection, DatabaseAccessDecision, NormalizedType,
    OperationKind, OperationNode, OrderTerm, ParameterSpec, QueryPlan,
    QueryPlanReference, QueryPlanRegistry, RawColumn, RawRelationalResult,
    VendorNeutralContract,
)


class RecordingAdapter:
    adapter_id = "memory"
    adapter_version = "1"

    def __init__(self, result):
        self.result = result
        self.calls = []

    def capabilities(self):
        return frozenset({"operation:select", "typed_parameters", "deterministic_order"})

    async def execute_read(self, plan, parameters, access, context):
        self.calls.append((plan, dict(parameters), access))
        return self.result

    async def cancel(self, cancellation_token):
        pass


class AllowPolicy:
    async def authorize_read(self, credential_id, plan, context):
        return DatabaseAccessDecision(True, credential_id, "reader-service", "security-1", "allow", "tls", "kms")


class RecordingAudit:
    def __init__(self):
        self.events = []

    async def emit(self, event):
        self.events.append(event)


def context():
    configuration = ExecutionConfiguration(
        ConfigurationReference("default", "v1"),
        freeze({"database": {"read_only_credential_id": "reader"}}),
        security_policy_version="security-1",
    )
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return ExecutionContext(ExecutionId("execution"), CorrelationId("correlation"), ExecutionPath.INTERACTIVE,
                            configuration, DeadlineContext(now, now), CancellationContext("cancel", 0))


def plan(operation=None, order=(OrderTerm("id"),)):
    return QueryPlan(QueryPlanReference("customers", "1"), operation or OperationNode(OperationKind.SELECT),
                     (ParameterSpec("minimum", NormalizedType.INTEGER),),
                     frozenset({"typed_parameters", "deterministic_order"}), order, False, 10, 1000,
                     frozenset({"customers"}))


def dal(query_plan, adapter, audit=None):
    return DataAccessLayer(QueryPlanRegistry((query_plan,)),
                           VendorNeutralContract("1", adapter.capabilities(), frozenset(NormalizedType)),
                           adapter, AllowPolicy(), audit or RecordingAudit(),
                           clock=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc))


def test_allowlisted_typed_read_is_submitted_and_normalized_in_declared_order():
    adapter = RecordingAdapter(RawRelationalResult(
        (RawColumn("id", NormalizedType.INTEGER, False), RawColumn("amount", NormalizedType.DECIMAL)),
        ((2, "2.00"), (1, Decimal("1.0"))),
    ))

    result = asyncio.run(dal(plan(), adapter).execute_read(QueryPlanReference("customers", "1"), {"minimum": 1}, context()))

    assert result.rows == ((1, Decimal("1")), (2, Decimal("2")))
    assert result.protection.transport_encryption == "tls"
    assert adapter.calls[0][1] == {"minimum": 1}


@pytest.mark.parametrize("query_plan,code", [
    (plan(order=()), FailureCode.NONDETERMINISTIC_ORDER),
    (plan(OperationNode(OperationKind.SELECT, (OperationNode(OperationKind.UPDATE),))), FailureCode.MUTATION_BLOCKED),
])
def test_unsafe_plan_is_rejected_before_adapter_submission(query_plan, code):
    adapter = RecordingAdapter(RawRelationalResult((), ()))
    audit = RecordingAudit()

    with pytest.raises(DataAccessRejection) as raised:
        asyncio.run(dal(query_plan, adapter, audit).execute_read(query_plan.reference, {"minimum": 1}, context()))

    assert raised.value.failure.code is code
    assert adapter.calls == []
    if code is FailureCode.MUTATION_BLOCKED:
        assert audit.events[0].redacted_details == "[REDACTED]"
        assert audit.events[0].correlation_id == "correlation"


def test_unallowlisted_plan_and_invalid_parameters_never_reach_adapter():
    adapter = RecordingAdapter(RawRelationalResult((), ()))
    layer = dal(plan(), adapter)

    with pytest.raises(DataAccessRejection) as missing:
        asyncio.run(layer.execute_read(QueryPlanReference("other", "1"), {}, context()))
    with pytest.raises(DataAccessRejection) as invalid:
        asyncio.run(layer.execute_read(QueryPlanReference("customers", "1"), {"minimum": "one"}, context()))

    assert missing.value.failure.code is FailureCode.QUERY_PLAN_UNAVAILABLE
    assert invalid.value.failure.code is FailureCode.INVALID_QUERY_PARAMETERS
    assert adapter.calls == []
