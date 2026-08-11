from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from ai_retrieval.admission.router import AdmissionRouter
from ai_retrieval.control_plane.configuration import StaticConfigurationBinder
from ai_retrieval.control_plane.context import DefaultExecutionContextFactory
from ai_retrieval.domain.admission import AdmissionEnvelope
from ai_retrieval.domain.configuration import ConfigurationReference
from ai_retrieval.domain.execution import ExecutionPath
from ai_retrieval.domain.failures import FailureCode
from ai_retrieval.domain.immutable import FrozenMapping
from ai_retrieval.domain.outcomes import ExecutionOutcome, OutcomeStatus
from ai_retrieval.domain.work import BatchJob, InteractiveRequest, QueryWork


class RecordingDispatcher:
    def __init__(self, component: str) -> None:
        self.component = component
        self.calls = []

    def dispatch(self, work, context):
        self.calls.append((work, context))
        return ExecutionOutcome(OutcomeStatus.ACCEPTED, self.component)


class RecordingBinder(StaticConfigurationBinder):
    def __init__(self, profiles):
        super().__init__(profiles)
        self.calls = 0

    def bind(self, reference):
        self.calls += 1
        return super().bind(reference)


@pytest.fixture
def components():
    reference = ConfigurationReference("default", "v1")
    source = {"nested": {"limit": 3}}
    binder = RecordingBinder({reference: source})
    interactive = RecordingDispatcher("interactive")
    bulk = RecordingDispatcher("bulk")
    identifiers = iter(("execution", "correlation", "cancel"))
    factory = DefaultExecutionContextFactory(
        clock=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc),
        identifier=lambda: next(identifiers),
    )
    router = AdmissionRouter(binder, factory, interactive, bulk)
    return reference, source, binder, interactive, bulk, router


def test_interactive_request_dispatches_only_to_interactive_path(components):
    reference, _, _, interactive, bulk, router = components
    request = InteractiveRequest("request-1", QueryWork("plan-1"))

    decision = router.admit(AdmissionEnvelope(reference, interactive=request, interactive_deadline_seconds=2))

    assert decision.path is ExecutionPath.INTERACTIVE
    assert len(interactive.calls) == 1
    assert bulk.calls == []
    assert decision.context.timing.deadline.isoformat() == "2025-01-01T00:00:02+00:00"


def test_bulk_job_dispatches_only_to_bulk_path_without_interactive_deadline(components):
    reference, _, _, interactive, bulk, router = components

    decision = router.admit(AdmissionEnvelope(reference, bulk=BatchJob("job-1", ("item-1",))))

    assert decision.path is ExecutionPath.BULK
    assert len(bulk.calls) == 1
    assert interactive.calls == []
    assert decision.context.timing.deadline is None


def test_discriminator_resolves_both_indicators_exclusively(components):
    reference, _, _, interactive, bulk, router = components
    envelope = AdmissionEnvelope(
        reference,
        interactive=InteractiveRequest("request-1", QueryWork("plan-1")),
        bulk=BatchJob("job-1", ("item-1",)),
        discriminator=ExecutionPath.BULK,
    )

    decision = router.admit(envelope)

    assert decision.path is ExecutionPath.BULK
    assert len(bulk.calls) == 1
    assert interactive.calls == []


def test_ambiguous_input_is_rejected_before_binding_or_dispatch(components):
    reference, _, binder, interactive, bulk, router = components
    envelope = AdmissionEnvelope(
        reference,
        interactive=InteractiveRequest("request-1", QueryWork("plan-1")),
        bulk=BatchJob("job-1", ("item-1",)),
    )

    decision = router.admit(envelope)

    assert decision.status is OutcomeStatus.REJECTED
    assert decision.failure.code is FailureCode.AMBIGUOUS_PATH
    assert decision.path is None and decision.context is None
    assert binder.calls == 0
    assert interactive.calls == [] and bulk.calls == []


def test_configuration_binding_is_deeply_immutable_and_detached(components):
    reference, source, _, _, _, router = components
    decision = router.admit(
        AdmissionEnvelope(reference, interactive=InteractiveRequest("request-1", QueryWork("plan-1")))
    )
    bound = decision.context.configuration

    source["nested"]["limit"] = 99

    assert bound.content["nested"]["limit"] == 3
    with pytest.raises((FrozenInstanceError, TypeError)):
        bound.content._items = ()
