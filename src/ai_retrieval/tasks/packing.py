"""Canonical provider-neutral payload serialization and bounded row packing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Protocol

from ai_retrieval.domain.execution import ExecutionContext
from ai_retrieval.domain.immutable import FrozenMapping, freeze
from ai_retrieval.domain.work import ModelWork
from ai_retrieval.tasks.models import (
    PackedRequest,
    PreparedTask,
    RowInput,
    TaskDefinition,
    TaskFailure,
    TaskFailureCode,
    TaskFailureStage,
)
from ai_retrieval.tasks.runtime import ResolvedTask


class TokenEstimator(Protocol):
    """Deterministic token estimates for complete serialized payloads."""

    def estimate_input(self, payload: bytes) -> int: ...

    def estimate_output(self, definition: TaskDefinition, row_count: int) -> int: ...


class PayloadReferenceStore(Protocol):
    """Content-addressed payload persistence boundary."""

    def put(self, payload: bytes, content_hash: str) -> str: ...


class CanonicalSerializationError(ValueError):
    """A value cannot be represented by the canonical JSON contract."""


@dataclass(frozen=True)
class _CandidatePack:
    rows: tuple[RowInput, ...]
    payload: bytes
    input_tokens: int
    output_tokens: int


def canonical_payload_bytes(value: object) -> bytes:
    """Return canonical UTF-8 JSON, rejecting unsupported and non-finite values."""
    try:
        return json.dumps(
            _canonical_plain(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CanonicalSerializationError(str(error)) from error


class DeterministicTaskPacker:
    """Prepare deterministic contiguous greedy packs after task resolution."""

    def __init__(
        self, estimator: TokenEstimator, payload_store: PayloadReferenceStore
    ) -> None:
        self._estimator = estimator
        self._payload_store = payload_store

    def prepare(
        self, resolved: ResolvedTask, context: ExecutionContext
    ) -> PreparedTask | TaskFailure:
        """Pack all rows before persisting any payload reference."""
        del context  # Resolution owns context concerns; packing is deterministic.
        definition = resolved.definition
        invocation = resolved.invocation
        try:
            candidates = self._build_candidates(resolved)
        except _OversizedRow as error:
            return _packing_failure(
                TaskFailureCode.OVERSIZED_ROW,
                "task_payload.row_fits_empty_pack",
                "oversized_row",
                row_id=error.row_id,
            )
        except CanonicalSerializationError as error:
            return _packing_failure(
                TaskFailureCode.INVALID_TASK_INVOCATION,
                "task_payload.canonical_json",
                "unsupported_canonical_value",
                error_type=type(error).__name__,
            )
        except (TypeError, ValueError) as error:
            return _packing_failure(
                TaskFailureCode.INVALID_TASK_INVOCATION,
                "task_payload.estimate",
                "invalid_token_estimate",
                error_type=type(error).__name__,
            )

        packs: list[PackedRequest] = []
        work: list[ModelWork] = []
        for index, candidate in enumerate(candidates):
            content_hash = sha256(candidate.payload).hexdigest()
            reference = self._payload_store.put(candidate.payload, content_hash)
            input_ids = tuple(row.identifier for row in candidate.rows)
            pack = PackedRequest(
                definition.function,
                definition.version,
                index,
                input_ids,
                candidate.payload,
                candidate.input_tokens,
                candidate.output_tokens,
            )
            packs.append(pack)
            work.append(
                ModelWork(
                    definition.function.value,
                    reference,
                    input_ids,
                    definition.required_capabilities,
                    definition.version,
                    candidate.input_tokens,
                    candidate.output_tokens,
                )
            )
        return PreparedTask(definition, invocation, tuple(packs), tuple(work))

    def _build_candidates(self, resolved: ResolvedTask) -> tuple[_CandidatePack, ...]:
        definition = resolved.definition
        invocation = resolved.invocation
        accepted: list[_CandidatePack] = []
        current_rows: tuple[RowInput, ...] = ()

        for row in invocation.rows:
            proposed = self._candidate(definition, invocation.parameters, current_rows + (row,))
            if self._fits(definition, proposed):
                current_rows = proposed.rows
                continue
            if not current_rows:
                raise _OversizedRow(row.identifier)
            accepted.append(self._candidate(definition, invocation.parameters, current_rows))
            single = self._candidate(definition, invocation.parameters, (row,))
            if not self._fits(definition, single):
                raise _OversizedRow(row.identifier)
            current_rows = single.rows

        if current_rows:
            accepted.append(self._candidate(definition, invocation.parameters, current_rows))
        return tuple(accepted)

    def _candidate(
        self,
        definition: TaskDefinition,
        parameters: FrozenMapping,
        rows: tuple[RowInput, ...],
    ) -> _CandidatePack:
        payload = canonical_payload_bytes(_payload_object(definition, parameters, rows))
        input_tokens = _estimate(
            self._estimator.estimate_input(payload), "input token estimate"
        )
        output_tokens = _estimate(
            self._estimator.estimate_output(definition, len(rows)),
            "output token estimate",
        )
        return _CandidatePack(rows, payload, input_tokens, output_tokens)

    @staticmethod
    def _fits(definition: TaskDefinition, candidate: _CandidatePack) -> bool:
        limits = definition.packing_limits
        return (
            len(candidate.rows) <= limits.max_items
            and candidate.input_tokens <= limits.max_input_tokens
            and candidate.output_tokens <= limits.max_output_tokens
            and len(candidate.payload) <= limits.max_payload_bytes
        )


class _OversizedRow(ValueError):
    def __init__(self, row_id: str) -> None:
        self.row_id = row_id
        super().__init__("row does not fit an empty pack")


def _payload_object(
    definition: TaskDefinition,
    parameters: FrozenMapping,
    rows: Sequence[RowInput],
) -> dict[str, object]:
    return {
        "task": definition.function.value,
        "definition_version": definition.version,
        "instructions": definition.prompt_template,
        "parameters": parameters,
        "rows": [
            {"id": row.identifier, "source": row.source_fields}
            for row in rows
        ],
        "output_schema": definition.output_schema,
    }


def _canonical_plain(value: object) -> object:
    if isinstance(value, FrozenMapping):
        return {key: _canonical_plain(item) for key, item in value.items()}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("canonical JSON object keys must be strings")
        return {key: _canonical_plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical_plain(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("canonical JSON numbers must be finite")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def _estimate(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _packing_failure(
    code: TaskFailureCode,
    rule_id: str,
    reason: str,
    **details: object,
) -> TaskFailure:
    immutable_details = freeze({"reason": reason, **details})
    assert isinstance(immutable_details, FrozenMapping)
    return TaskFailure(
        TaskFailureStage.PACKING,
        code,
        (rule_id,),
        immutable_details,
    )
