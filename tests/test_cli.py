import json

import pytest

from ai_retrieval import composition
from ai_retrieval.cli.app import main
from ai_retrieval.domain.immutable import FrozenMapping
from ai_retrieval.tasks import TaskFunction


def _capture_task_admission(monkeypatch, path, captured):
    original_builder = composition.build_prototype

    def build_capturing_prototype(**kwargs):
        application = original_builder(**kwargs)
        method_name = f"admit_{path}_task"
        original_admit = getattr(application, method_name)

        def capture(*args, **admit_kwargs):
            captured["arguments"] = args
            captured["keywords"] = admit_kwargs
            return original_admit(*args, **admit_kwargs)

        setattr(application, method_name, capture)
        return application

    monkeypatch.setattr(composition, "build_prototype", build_capturing_prototype)


@pytest.mark.parametrize(
    "path,task_arguments,expected_function,expected_parameters,expected_records",
    (
        (
            "interactive", ("--label", "priority", "--label", "secondary"),
            TaskFunction.CLASSIFY, {"labels": ("priority", "secondary")},
            [{"id": "customer-1", "label": "priority"}],
        ),
        (
            "interactive", ("--max-words", "2"),
            TaskFunction.SUMMARIZE, {"max_words": 2},
            [{"id": "customer-1", "summary": "Ada"}],
        ),
        (
            "bulk", ("--label", "priority", "--label", "secondary"),
            TaskFunction.CLASSIFY, {"labels": ("priority", "secondary")},
            [
                {"id": "item-1", "label": "priority"},
                {"id": "item-2", "label": "priority"},
            ],
        ),
        (
            "bulk", ("--max-words", "2"),
            TaskFunction.SUMMARIZE, {"max_words": 2},
            [
                {"id": "item-1", "summary": "item-1"},
                {"id": "item-2", "summary": "item-2"},
            ],
        ),
    ),
)
def test_task_cli_constructs_immutable_invocation_and_preserves_output_order(
    capsys, monkeypatch, path, task_arguments, expected_function,
    expected_parameters, expected_records,
):
    captured = {}
    _capture_task_admission(monkeypatch, path, captured)
    if path == "interactive":
        command = [
            "interactive", "--request-id", "task-request", "--query-plan", "customers",
            "--config-version", "ignored-for-prototype", "--execute",
            "--task", expected_function.value, *task_arguments,
        ]
    else:
        command = [
            "bulk", "--job-id", "ordered-job", "--item", "item-2", "--item", "item-1",
            "--config-version", "ignored-for-prototype", "--execute",
            "--task", expected_function.value, *task_arguments,
        ]

    exit_code = main(command)
    output = json.loads(capsys.readouterr().out)

    function_index, parameters_index = ((3, 4) if path == "interactive" else (2, 3))
    parameters = captured["arguments"][parameters_index]
    assert exit_code == 0
    assert captured["arguments"][function_index] is expected_function
    if path == "bulk":
        assert captured["arguments"][1] == ("item-2", "item-1")
    assert isinstance(parameters, FrozenMapping)
    assert dict(parameters) == expected_parameters
    assert captured["keywords"] == {}
    with pytest.raises(TypeError):
        parameters["changed"] = True
    assert output["status"] == "accepted"
    assert output["path"] == path
    assert output["outcome"]["task"] == expected_function.value
    assert output["outcome"]["task_failures"] == []
    if path == "interactive":
        task_results = output["outcome"]["task_results"]
        assert len(task_results) == 1
        records = task_results[0]["records"]
        assert task_results[0]["operation_id"].startswith(
            f"task:{expected_function.value}:"
        )
        assert output["outcome"]["read_only"] is True
    else:
        records = [
            record
            for item_result in output["outcome"]["task_results"]
            for record in item_result["records"]
        ]
        assert [item["item_id"] for item in output["outcome"]["task_results"]] == [
            "item-1", "item-2",
        ]
    assert records == expected_records


@pytest.mark.parametrize(
    "task_arguments,expected_task,expected_record",
    (
        (("--task", "ai_classify", "--label", "ok"), "ai_classify", {"id": "item-1", "label": "ok"}),
        (("--task", "ai_summarize", "--max-words", "1"), "ai_summarize", {"id": "item-1", "summary": "item-1"}),
    ),
)
@pytest.mark.parametrize(
    "write_mode,expected_status,expected_mutations",
    (("disabled", "persisted", 0), ("approved", "committed", 1)),
)
def test_task_bulk_cli_reports_terminal_status_and_write_boundary(
    capsys, task_arguments, expected_task, expected_record,
    write_mode, expected_status, expected_mutations,
):
    exit_code = main([
        "bulk", "--job-id", "bulk-job", "--item", "item-1",
        "--config-version", "ignored-for-prototype", "--execute", "--resume",
        "--write-back", write_mode, *task_arguments,
    ])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["outcome"]["task"] == expected_task
    assert output["outcome"]["classification"] == "succeeded"
    assert output["outcome"]["states"] == {"item-1": "succeeded"}
    assert output["outcome"]["task_results"] == [
        {"item_id": "item-1", "records": [expected_record]},
    ]
    assert output["outcome"]["task_failures"] == []
    assert output["outcome"]["write_back"] == {"item-1": expected_status}
    assert output["outcome"]["mutation_count"] == expected_mutations
    assert output["outcome"]["resumed"] is True


@pytest.mark.parametrize(
    "path,task,expected_rule",
    (
        ("interactive", "ai_classify", "task_invocation.classify.labels.nonempty"),
        ("interactive", "ai_summarize", "task_invocation.summarize.max_words.positive_integer"),
        ("bulk", "ai_classify", "task_invocation.classify.labels.nonempty"),
        ("bulk", "ai_summarize", "task_invocation.summarize.max_words.positive_integer"),
    ),
)
def test_task_cli_validation_failures_are_stable_typed_and_nonzero(
    capsys, path, task, expected_rule,
):
    if path == "interactive":
        command = [
            "interactive", "--request-id", "invalid-task", "--query-plan", "customers",
            "--config-version", "ignored-for-prototype", "--execute", "--task", task,
        ]
    else:
        command = [
            "bulk", "--job-id", "invalid-job", "--item", "item-1",
            "--config-version", "ignored-for-prototype", "--execute", "--task", task,
        ]

    first_code = main(command)
    first = json.loads(capsys.readouterr().out)
    second_code = main(command)
    second = json.loads(capsys.readouterr().out)

    assert first_code == second_code == 2
    assert first["status"] == second["status"] == "accepted"
    assert first["outcome"]["task_results"] == second["outcome"]["task_results"] == []
    assert first["outcome"]["task_failures"] == second["outcome"]["task_failures"]
    failure = first["outcome"]["task_failures"][0]
    if path == "interactive":
        assert first["outcome"]["complete"] is False
        assert first["outcome"]["terminal_reason"] == "validation_failed"
        assert failure["operation_id"] == f"task:{task}:prepare"
        assert failure["reason"] == "validation_failed"
        assert failure["details"]["failure_code"] == "invalid_task_invocation"
        assert expected_rule in failure["details"]["failed_rule_ids"]
    else:
        assert first["outcome"]["classification"] == "failed"
        assert first["outcome"]["states"] == {"item-1": "validation-failed"}
        assert failure["item_id"] == "item-1"
        assert failure["state"] == "validation-failed"
        assert failure["code"] == "invalid_task_invocation"
        assert expected_rule in failure["details"]["rules"]
        assert first["outcome"]["write_back"] == {}
        assert first["outcome"]["mutation_count"] == 0


@pytest.mark.parametrize(
    "task,valid_parameters",
    (
        ("ai_classify", ("--label", "ok")),
        ("ai_summarize", ("--max-words", "1")),
    ),
)
def test_interactive_task_cli_output_validation_failure_uses_existing_typed_contract(
    capsys, task, valid_parameters,
):
    exit_code = main([
        "interactive", "--request-id", "invalid-output", "--query-plan", "customers",
        "--config-version", "ignored-for-prototype", "--execute", "--fallback",
        "--task", task, *valid_parameters,
    ])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert output["outcome"]["task_results"] == []
    failure = output["outcome"]["task_failures"][0]
    assert output["outcome"]["complete"] is False
    assert output["outcome"]["terminal_reason"] == "validation_failed"
    assert failure["reason"] == "validation_failed"
    assert failure["details"]["failure_code"] == "output_schema_violation"
    assert output["outcome"]["read_only"] is True


@pytest.mark.parametrize(
    "command,expected_path,expected_legacy_keys",
    (
        (
            ["interactive", "--request-id", "legacy-admission", "--query-plan", "customers", "--config-version", "v1"],
            "interactive", {"query_plan", "request_id"},
        ),
        (
            ["bulk", "--job-id", "legacy-admission", "--item", "item-1", "--config-version", "v1"],
            "bulk", {"item_count", "job_id"},
        ),
        (
            ["interactive", "--request-id", "legacy-prototype", "--query-plan", "customers", "--config-version", "v1", "--execute"],
            "interactive", {"request_id", "complete", "terminal_reason", "result_operations", "incomplete_operations", "read_only", "telemetry_count"},
        ),
        (
            ["bulk", "--job-id", "legacy-prototype", "--item", "item-1", "--config-version", "v1", "--execute"],
            "bulk", {"job_id", "classification", "states", "attempts", "write_back", "mutation_count", "resumed", "telemetry_count"},
        ),
    ),
)
def test_no_task_cli_keeps_legacy_admission_and_prototype_shapes(
    capsys, command, expected_path, expected_legacy_keys,
):
    exit_code = main(command)
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["status"] == "accepted"
    assert output["path"] == expected_path
    assert set(output["outcome"]) == expected_legacy_keys
    assert "task" not in output["outcome"]
    assert "task_results" not in output["outcome"]
    assert "task_failures" not in output["outcome"]


def test_interactive_cli_invokes_available_admission_behavior(capsys):
    exit_code = main(
        [
            "interactive",
            "--request-id",
            "request-1",
            "--query-plan",
            "customers",
            "--config-version",
            "v1",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["status"] == "accepted"
    assert output["path"] == "interactive"
    assert output["configuration_version"] == "v1"
    assert output["outcome"] == {"query_plan": "customers", "request_id": "request-1"}


def test_bulk_cli_accepts_repeated_items(capsys):
    exit_code = main(
        [
            "bulk",
            "--job-id",
            "job-1",
            "--item",
            "item-1",
            "--item",
            "item-2",
            "--config-version",
            "v2",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["path"] == "bulk"
    assert output["outcome"] == {"item_count": 2, "job_id": "job-1"}


def test_interactive_cli_execute_exercises_completion_and_fallback(capsys):
    completed_code = main([
        "interactive", "--request-id", "request-complete", "--query-plan", "customers",
        "--config-version", "ignored-for-prototype", "--execute",
    ])
    completed = json.loads(capsys.readouterr().out)

    fallback_code = main([
        "interactive", "--request-id", "request-fallback", "--query-plan", "customers",
        "--config-version", "ignored-for-prototype", "--execute", "--fallback",
    ])
    fallback = json.loads(capsys.readouterr().out)

    assert completed_code == fallback_code == 0
    assert completed["outcome"]["complete"] is True
    assert completed["outcome"]["terminal_reason"] == "complete"
    assert completed["outcome"]["read_only"] is True
    assert fallback["outcome"]["complete"] is False
    assert fallback["outcome"]["terminal_reason"] == "validation_failed"


def test_bulk_cli_execute_exercises_resume_and_write_boundaries(capsys):
    disabled_code = main([
        "bulk", "--job-id", "bulk-job", "--item", "item-1", "--config-version", "legacy-v1",
        "--execute", "--resume", "--write-back", "disabled",
    ])
    disabled = json.loads(capsys.readouterr().out)

    approved_code = main([
        "bulk", "--job-id", "bulk-job", "--item", "item-1", "--config-version", "legacy-v1",
        "--execute", "--write-back", "approved",
    ])
    approved = json.loads(capsys.readouterr().out)

    assert disabled_code == approved_code == 0
    assert disabled["outcome"]["classification"] == "succeeded"
    assert disabled["outcome"]["write_back"] == {"item-1": "persisted"}
    assert disabled["outcome"]["mutation_count"] == 0
    assert disabled["outcome"]["resumed"] is True
    assert approved["outcome"]["write_back"] == {"item-1": "committed"}
    assert approved["outcome"]["mutation_count"] == 1
