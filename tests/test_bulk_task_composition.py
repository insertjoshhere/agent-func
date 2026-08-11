import json

from ai_retrieval.composition import build_prototype
from ai_retrieval.domain.immutable import freeze
from ai_retrieval.tasks import TaskFunction


def test_bulk_task_persists_metadata_routes_parses_validates_and_resumes_without_reinvoke():
    app = build_prototype(write_back_enabled=False)
    decision = app.admit_bulk_task(
        "bulk-job", ("item-1",), TaskFunction.CLASSIFY,
        freeze({"labels": ("approved", "rejected")}),
        source_rows=freeze({"item-1": {"text": "customer text"}}),
    )

    app.worker.resume("bulk-job", "bulk-job:item-1", "worker-1", decision.context, max_stages=2)
    assert app.model_provider.calls == 1
    accepted_checkpoint = app.repository.checkpoints("bulk-job", "bulk-job:item-1")[0]
    manifest = json.loads(app.objects.get(accepted_checkpoint.result_reference))
    assert manifest["function"] == "ai_classify"
    assert manifest["definition_version"]
    assert manifest["definition_hash"]
    assert manifest["packs"][0]["payload_reference"].startswith("memory://payload/")

    result = app.run_bulk(decision)
    assert app.model_provider.calls == 1
    assert result.classification == "succeeded"
    assert dict(result.states) == {"item-1": "succeeded"}
    terminal = app.repository.terminal_items("bulk-job")[0]
    accepted = json.loads(app.objects.get(terminal.result_reference))
    assert accepted == [{"id": "item-1", "label": "approved"}]
    assert result.mutation_count == 0


def test_bulk_task_failure_isolated_and_terminal_classification_preserved():
    app = build_prototype(write_back_enabled=False)
    app.model_provider.invalid_item_ids.add("item-2")
    decision = app.admit_bulk_task(
        "bulk-job", ("item-1", "item-2"), TaskFunction.SUMMARIZE,
        freeze({"max_words": 2}),
        source_rows=freeze({
            "item-1": {"text": "one two three"},
            "item-2": {"text": "four five six"},
        }),
    )

    result = app.run_bulk(decision)

    assert result.classification == "partially-succeeded"
    assert dict(result.states) == {
        "item-1": "succeeded",
        "item-2": "validation-failed",
    }
    assert dict(result.write_back_statuses) == {"item-1": "persisted"}
    assert result.mutation_count == 0


def test_bulk_task_approved_write_back_receives_only_canonical_accepted_output():
    app = build_prototype(write_back_enabled=True)
    decision = app.admit_bulk_task(
        "bulk-job", ("item-1",), TaskFunction.CLASSIFY,
        freeze({"labels": ("ok",)}),
        source_rows=freeze({"item-1": {"text": "customer text"}}),
    )

    result = app.run_bulk(decision, simulate_interrupt=True)

    assert result.classification == "succeeded"
    assert dict(result.write_back_statuses) == {"item-1": "committed"}
    assert result.mutation_count == 1
    assert app.model_provider.calls == 1
    effect = app.repository.effect("bulk-job", "bulk-job:item-1")
    assert effect is not None
