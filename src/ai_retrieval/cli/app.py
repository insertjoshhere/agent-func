"""Simple CLI composition for admission behavior available in this task."""

import argparse
import json
from typing import Any

from ai_retrieval.admission.router import AdmissionRouter
from ai_retrieval.control_plane.configuration import StaticConfigurationBinder
from ai_retrieval.control_plane.context import DefaultExecutionContextFactory
from ai_retrieval.domain.admission import AdmissionEnvelope
from ai_retrieval.domain.configuration import ConfigurationReference
from ai_retrieval.domain.execution import ExecutionContext
from ai_retrieval.domain.immutable import FrozenMapping, freeze
from ai_retrieval.domain.outcomes import ExecutionOutcome, OutcomeStatus
from ai_retrieval.domain.work import BatchJob, InteractiveRequest, QueryWork


class CliInteractiveDispatcher:
    def dispatch(self, request: InteractiveRequest, context: ExecutionContext) -> ExecutionOutcome:
        return ExecutionOutcome(
            OutcomeStatus.ACCEPTED,
            "interactive",
            FrozenMapping((("request_id", request.request_id), ("query_plan", request.query.plan_id))),
        )


class CliBulkDispatcher:
    def dispatch(self, job: BatchJob, context: ExecutionContext) -> ExecutionOutcome:
        return ExecutionOutcome(
            OutcomeStatus.ACCEPTED,
            "bulk",
            FrozenMapping((("item_count", len(job.item_ids)), ("job_id", job.job_id))),
        )


def _add_task_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--task", choices=("ai_classify", "ai_summarize"),
        help="execute a provider-neutral task through the selected path",
    )
    parser.add_argument(
        "--label", action="append", dest="labels",
        help="classification label; repeat to preserve label order",
    )
    parser.add_argument(
        "--max-words",
        help="positive summary word limit",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exercise exclusive AI retrieval admission")
    subparsers = parser.add_subparsers(dest="path", required=True)

    interactive = subparsers.add_parser("interactive", help="admit a read-only interactive request")
    interactive.add_argument("--request-id", required=True)
    interactive.add_argument("--query-plan", required=True)
    interactive.add_argument("--config-version", required=True)
    interactive.add_argument("--deadline-seconds", type=float, default=2.0)
    interactive.add_argument(
        "--execute", action="store_true",
        help="run the composed read-only adapter/model/validation path",
    )
    interactive.add_argument(
        "--fallback", action="store_true",
        help="make the fake model output fail deterministic validation",
    )
    _add_task_arguments(interactive)

    bulk = subparsers.add_parser("bulk", help="admit a bulk job")
    bulk.add_argument("--job-id", required=True)
    bulk.add_argument("--item", action="append", required=True, dest="items")
    bulk.add_argument("--config-version", required=True)
    bulk.add_argument(
        "--execute", action="store_true",
        help="run the composed durable coordinator/worker path",
    )
    bulk.add_argument(
        "--resume", action="store_true",
        help="simulate interruption after one checkpoint, then resume",
    )
    bulk.add_argument(
        "--write-back", choices=("disabled", "approved"), default="disabled",
        help="persist validated output only or use the approved fake effect adapter",
    )
    _add_task_arguments(bulk)
    return parser


def _serialize(value: Any) -> Any:
    if isinstance(value, FrozenMapping):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, frozenset):
        return sorted((_serialize(item) for item in value), key=repr)
    return value


def _task_parameters(args) -> FrozenMapping:
    if args.task == "ai_classify":
        values: dict[str, Any] = {"labels": tuple(args.labels or ())}
        if args.max_words is not None:
            values["max_words"] = _integer_or_original(args.max_words)
    else:
        values = {"max_words": _integer_or_original(args.max_words)}
        if args.labels:
            values["labels"] = tuple(args.labels)
    parameters = freeze(values)
    assert isinstance(parameters, FrozenMapping)
    return parameters


def _integer_or_original(value: str | None) -> int | str | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return value


def _task_interactive_outcome(application, response, task: str) -> dict[str, Any]:
    task_results = tuple(
        {
            "operation_id": item.operation_id,
            "records": _serialize(item.value),
        }
        for item in response.results
        if item.operation_id.startswith("task:")
    )
    task_failures = tuple(
        {
            "operation_id": item.operation_id,
            "reason": item.reason.value,
            "details": _serialize(item.details),
        }
        for item in response.incompleteness
        if item.operation_id.startswith("task:")
    )
    return {
        "request_id": response.request_id,
        "task": task,
        "complete": response.complete,
        "terminal_reason": response.terminal_reason.value,
        "task_results": task_results,
        "task_failures": task_failures,
        "read_only": application.effect_adapter.mutation_count == 0,
        "telemetry_count": len(application.telemetry.events),
    }


def _task_bulk_outcome(application, result, task: str, resumed: bool) -> dict[str, Any]:
    task_results: list[dict[str, Any]] = []
    task_failures: list[dict[str, Any]] = []
    items = {item.item_id: item for item in application.repository.items(result.job_id)}
    terminals = {
        item.item_id: item for item in application.repository.terminal_items(result.job_id)
    }
    for item_id, state in result.states:
        item = items[item_id]
        terminal = terminals.get(item_id)
        if state == "succeeded" and item.result_reference is not None:
            records = json.loads(application.objects.get(item.result_reference).decode("utf-8"))
            task_results.append({"item_id": item_id, "records": records})
        elif terminal is not None:
            task_failures.append({
                "item_id": item_id,
                "state": terminal.state.value,
                "code": terminal.failure_code,
                "details": _json_or_text(terminal.failure_details),
            })
    return {
        "job_id": result.job_id,
        "task": task,
        "classification": result.classification,
        "states": dict(result.states),
        "attempts": dict(result.attempts),
        "task_results": task_results,
        "task_failures": task_failures,
        "write_back": dict(result.write_back_statuses),
        "mutation_count": result.mutation_count,
        "resumed": resumed,
        "telemetry_count": result.telemetry_count,
    }


def _json_or_text(value: str | None) -> Any:
    if value is None:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _task_prototype_main(args, application) -> int:
    from ai_retrieval.tasks import TaskFunction

    function = TaskFunction(args.task)
    parameters = _task_parameters(args)
    if args.path == "interactive":
        decision = application.admit_interactive_task(
            args.request_id, args.query_plan, args.deadline_seconds,
            function, parameters,
        )
        response = decision.outcome
        outcome = (
            _task_interactive_outcome(application, response, function.value)
            if response is not None else None
        )
        valid = decision.status is OutcomeStatus.ACCEPTED and response.complete
    else:
        decision = application.admit_bulk_task(
            args.job_id, tuple(args.items), function, parameters,
        )
        result = application.run_bulk(decision, simulate_interrupt=args.resume)
        outcome = _task_bulk_outcome(application, result, function.value, args.resume)
        valid = (
            decision.status is OutcomeStatus.ACCEPTED
            and all(state == "succeeded" for _, state in result.states)
        )
    output = {
        "status": decision.status.value,
        "path": decision.path.value if decision.path else None,
        "execution_id": str(decision.context.execution_id) if decision.context else None,
        "correlation_id": str(decision.context.correlation_id) if decision.context else None,
        "configuration_version": decision.context.configuration.reference.version if decision.context else None,
        "outcome": outcome,
        "failure": decision.failure.code.value if decision.failure else None,
    }
    print(json.dumps(output, sort_keys=True))
    return 0 if valid else 2


def _prototype_main(args) -> int:
    from ai_retrieval.composition import build_prototype

    application = build_prototype(
        write_back_enabled=getattr(args, "write_back", "disabled") == "approved",
        model_valid=not getattr(args, "fallback", False),
    )
    if args.task is not None:
        return _task_prototype_main(args, application)
    if args.path == "interactive":
        decision = application.admit_interactive(
            args.request_id, args.query_plan, args.deadline_seconds,
        )
        response = decision.outcome
        output = {
            "status": decision.status.value,
            "path": decision.path.value if decision.path else None,
            "execution_id": str(decision.context.execution_id) if decision.context else None,
            "correlation_id": str(decision.context.correlation_id) if decision.context else None,
            "configuration_version": decision.context.configuration.reference.version if decision.context else None,
            "outcome": {
                "request_id": response.request_id,
                "complete": response.complete,
                "terminal_reason": response.terminal_reason.value,
                "result_operations": [item.operation_id for item in response.results],
                "incomplete_operations": [item.operation_id for item in response.incompleteness],
                "read_only": application.effect_adapter.mutation_count == 0,
                "telemetry_count": len(application.telemetry.events),
            },
            "failure": decision.failure.code.value if decision.failure else None,
        }
    else:
        decision = application.admit_bulk(args.job_id, tuple(args.items))
        result = application.run_bulk(decision, simulate_interrupt=args.resume)
        output = {
            "status": decision.status.value,
            "path": decision.path.value if decision.path else None,
            "execution_id": str(decision.context.execution_id) if decision.context else None,
            "correlation_id": str(decision.context.correlation_id) if decision.context else None,
            "configuration_version": decision.context.configuration.reference.version if decision.context else None,
            "outcome": {
                "job_id": result.job_id,
                "classification": result.classification,
                "states": dict(result.states),
                "attempts": dict(result.attempts),
                "write_back": dict(result.write_back_statuses),
                "mutation_count": result.mutation_count,
                "resumed": args.resume,
                "telemetry_count": result.telemetry_count,
            },
            "failure": decision.failure.code.value if decision.failure else None,
        }
    print(json.dumps(output, sort_keys=True))
    return 0 if decision.status is OutcomeStatus.ACCEPTED else 2


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.execute:
        return _prototype_main(args)
    reference = ConfigurationReference("cli", args.config_version)
    profile = freeze({"routing": {"interactive": True, "bulk": True}})
    assert isinstance(profile, FrozenMapping)
    router = AdmissionRouter(
        StaticConfigurationBinder({reference: profile}),
        DefaultExecutionContextFactory(),
        CliInteractiveDispatcher(),
        CliBulkDispatcher(),
    )

    if args.path == "interactive":
        envelope = AdmissionEnvelope(
            configuration=reference,
            interactive=InteractiveRequest(args.request_id, QueryWork(args.query_plan)),
            interactive_deadline_seconds=args.deadline_seconds,
        )
    else:
        envelope = AdmissionEnvelope(
            configuration=reference,
            bulk=BatchJob(args.job_id, tuple(args.items)),
        )

    decision = router.admit(envelope)
    output = {
        "status": decision.status.value,
        "path": decision.path.value if decision.path else None,
        "execution_id": str(decision.context.execution_id) if decision.context else None,
        "correlation_id": str(decision.context.correlation_id) if decision.context else None,
        "configuration_version": decision.context.configuration.reference.version if decision.context else None,
        "outcome": _serialize(decision.outcome.details) if decision.outcome else None,
        "failure": decision.failure.code.value if decision.failure else None,
    }
    print(json.dumps(output, sort_keys=True))
    return 0 if decision.status is OutcomeStatus.ACCEPTED else 2
