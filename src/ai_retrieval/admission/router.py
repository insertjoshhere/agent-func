"""Exclusive admission router with rejection before configuration or dispatch effects."""

from typing import Generic, TypeVar

from ai_retrieval.bulk.ports import BulkDispatcher
from ai_retrieval.control_plane.configuration import ConfigurationBinder, ConfigurationBindingError
from ai_retrieval.control_plane.context import ExecutionContextFactory
from ai_retrieval.domain.admission import AdmissionEnvelope
from ai_retrieval.domain.execution import ExecutionPath
from ai_retrieval.domain.failures import FailureCode, TypedFailure
from ai_retrieval.domain.outcomes import AdmissionDecision, OutcomeStatus
from ai_retrieval.interactive.ports import InteractiveDispatcher


T = TypeVar("T")


def discriminate_path(envelope: AdmissionEnvelope) -> ExecutionPath | TypedFailure:
    """Resolve exactly one supplied payload or return a typed routing failure."""
    has_interactive = envelope.interactive is not None
    has_bulk = envelope.bulk is not None

    if has_interactive and has_bulk:
        if envelope.discriminator is not None:
            return envelope.discriminator
        return TypedFailure(FailureCode.AMBIGUOUS_PATH, "both path indicators require one discriminator")

    if has_interactive:
        if envelope.discriminator in (None, ExecutionPath.INTERACTIVE):
            return ExecutionPath.INTERACTIVE
        return TypedFailure(FailureCode.INVALID_DISCRIMINATOR, "bulk discriminator has no bulk payload")

    if has_bulk:
        if envelope.discriminator in (None, ExecutionPath.BULK):
            return ExecutionPath.BULK
        return TypedFailure(FailureCode.INVALID_DISCRIMINATOR, "interactive discriminator has no interactive payload")

    return TypedFailure(FailureCode.MISSING_PATH, "one interactive or bulk payload is required")


class AdmissionRouter(Generic[T]):
    def __init__(
        self,
        configuration_binder: ConfigurationBinder,
        context_factory: ExecutionContextFactory,
        interactive_dispatcher: InteractiveDispatcher[T],
        bulk_dispatcher: BulkDispatcher[T],
    ) -> None:
        self._configuration_binder = configuration_binder
        self._context_factory = context_factory
        self._interactive_dispatcher = interactive_dispatcher
        self._bulk_dispatcher = bulk_dispatcher

    def admit(self, envelope: AdmissionEnvelope) -> AdmissionDecision[T]:
        resolved = discriminate_path(envelope)
        if isinstance(resolved, TypedFailure):
            return self._rejected(resolved)

        try:
            configuration = self._configuration_binder.bind(envelope.configuration)
        except ConfigurationBindingError as error:
            return self._rejected(error.failure)

        deadline_seconds = envelope.interactive_deadline_seconds if resolved is ExecutionPath.INTERACTIVE else None
        context = self._context_factory.create(
            resolved,
            configuration,
            deadline_seconds,
            envelope.cancellation_timeout_seconds,
        )

        if resolved is ExecutionPath.INTERACTIVE:
            assert envelope.interactive is not None
            outcome = self._interactive_dispatcher.dispatch(envelope.interactive, context)
        else:
            assert envelope.bulk is not None
            outcome = self._bulk_dispatcher.dispatch(envelope.bulk, context)

        return AdmissionDecision(OutcomeStatus.ACCEPTED, resolved, context, outcome, None)

    @staticmethod
    def _rejected(failure: TypedFailure) -> AdmissionDecision[T]:
        return AdmissionDecision(OutcomeStatus.REJECTED, None, None, None, failure)
