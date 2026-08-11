"""Strict provider-neutral parsing for supported structured task output."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
import math
from typing import TypeAlias

from ai_retrieval.domain.immutable import FrozenMapping, freeze
from ai_retrieval.tasks.models import (
    TaskDefinition,
    TaskFailureCode,
    TaskFailureStage,
    TaskFunction,
    TaskOutputFailure,
)

EnvelopeExtractor: TypeAlias = Callable[[object], object]
ParsedTaskRecords: TypeAlias = tuple[FrozenMapping, ...]
ParseResult: TypeAlias = ParsedTaskRecords | TaskOutputFailure


class _DuplicateKeyError(ValueError):
    pass


class _NonFiniteNumberError(ValueError):
    pass


class _UnsupportedValueError(TypeError):
    pass


def _object_from_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError
        result[key] = value
    return result


def _reject_constant(_: str) -> object:
    raise _NonFiniteNumberError


def _representation(value: object) -> str:
    if isinstance(value, bytes):
        return "bytes"
    if isinstance(value, str):
        return "text"
    if isinstance(value, Mapping):
        return "mapping"
    if isinstance(value, Sequence):
        return "sequence"
    return "unsupported"


def _normalize(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _NonFiniteNumberError
        return value
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise _UnsupportedValueError
            if key in result:
                raise _DuplicateKeyError
            result[key] = _normalize(item)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize(item) for item in value]
    raise _UnsupportedValueError


def _schema_fields(definition: TaskDefinition) -> tuple[str, str]:
    if definition.function is TaskFunction.CLASSIFY:
        return "id", "label"
    return "id", "summary"


class StructuredOutputParser:
    """Parse strict JSON-compatible responses using only configured envelopes."""

    def __init__(self, envelope_extractors: Mapping[str, EnvelopeExtractor] | None = None) -> None:
        extractors = {} if envelope_extractors is None else dict(envelope_extractors)
        if any(not isinstance(name, str) or not name.strip() for name in extractors):
            raise ValueError("envelope extractor names must be non-blank strings")
        if any(not callable(extractor) for extractor in extractors.values()):
            raise TypeError("envelope extractors must be callable")
        self._envelope_extractors = extractors

    def parse(
        self,
        response: object,
        definition: TaskDefinition,
        *,
        envelope: str | None = None,
    ) -> ParseResult:
        """Return immutable records or a stable failure without raw response data."""
        representation = _representation(response)
        if envelope is not None:
            extractor = self._envelope_extractors.get(envelope)
            if extractor is None:
                return self._failure(
                    TaskFailureCode.UNSUPPORTED_REPRESENTATION,
                    "structured_output.representation",
                    representation="envelope",
                    envelope_configured=False,
                )
            try:
                response = extractor(response)
            except Exception:
                return self._failure(
                    TaskFailureCode.MALFORMED_STRUCTURED_OUTPUT,
                    "structured_output.envelope",
                    representation="envelope",
                    envelope_configured=True,
                )
            representation = f"envelope:{_representation(response)}"

        decoded = self._decode(response, definition, representation)
        if isinstance(decoded, TaskOutputFailure):
            return decoded
        return self._validate_records(decoded, definition, representation)

    def _decode(
        self,
        response: object,
        definition: TaskDefinition,
        representation: str,
    ) -> object | TaskOutputFailure:
        if isinstance(response, bytes):
            if len(response) > definition.response_size_limit:
                return self._too_large(representation, len(response), definition.response_size_limit)
            try:
                text = response.decode("utf-8")
            except UnicodeDecodeError:
                return self._malformed(representation, "invalid_utf8")
            return self._decode_json(text, representation)

        if isinstance(response, str):
            size = len(response.encode("utf-8"))
            if size > definition.response_size_limit:
                return self._too_large(representation, size, definition.response_size_limit)
            return self._decode_json(response, representation)

        if isinstance(response, bytearray):
            return self._failure(
                TaskFailureCode.UNSUPPORTED_REPRESENTATION,
                "structured_output.representation",
                representation=representation,
            )

        if isinstance(response, (Mapping, Sequence)):
            if isinstance(response, (str, bytes, bytearray)):
                raise AssertionError("text and binary representations must be handled first")
            try:
                return _normalize(response)
            except _NonFiniteNumberError:
                return self._malformed(representation, "non_finite_number")
            except _DuplicateKeyError:
                return self._malformed(representation, "duplicate_object_key")
            except _UnsupportedValueError:
                return self._failure(
                    TaskFailureCode.UNSUPPORTED_REPRESENTATION,
                    "structured_output.representation",
                    representation=representation,
                )

        return self._failure(
            TaskFailureCode.UNSUPPORTED_REPRESENTATION,
            "structured_output.representation",
            representation=representation,
        )

    def _decode_json(self, text: str, representation: str) -> object | TaskOutputFailure:
        try:
            return json.loads(
                text,
                object_pairs_hook=_object_from_pairs,
                parse_constant=_reject_constant,
            )
        except _DuplicateKeyError:
            return self._malformed(representation, "duplicate_object_key")
        except _NonFiniteNumberError:
            return self._malformed(representation, "non_finite_number")
        except (json.JSONDecodeError, RecursionError):
            return self._malformed(representation, "invalid_json")

    def _validate_records(
        self,
        value: object,
        definition: TaskDefinition,
        representation: str,
    ) -> ParseResult:
        if not isinstance(value, list):
            return self._schema_failure(representation, "array_root_required")

        required_fields = _schema_fields(definition)
        expected = set(required_fields)
        records: list[FrozenMapping] = []
        for index, record in enumerate(value):
            if not isinstance(record, Mapping):
                return self._schema_failure(representation, "record_object_required", index)
            if set(record) != expected:
                return self._schema_failure(representation, "exact_fields_required", index)
            if any(not isinstance(record[field], str) for field in required_fields):
                return self._schema_failure(representation, "string_fields_required", index)
            frozen = freeze(record)
            assert isinstance(frozen, FrozenMapping)
            records.append(frozen)
        return tuple(records)

    @staticmethod
    def _failure(
        code: TaskFailureCode,
        rule_id: str,
        **details: object,
    ) -> TaskOutputFailure:
        frozen_details = freeze(details)
        assert isinstance(frozen_details, FrozenMapping)
        return TaskOutputFailure(
            TaskFailureStage.PARSING,
            code,
            (rule_id,),
            frozen_details,
        )

    def _malformed(self, representation: str, reason: str) -> TaskOutputFailure:
        return self._failure(
            TaskFailureCode.MALFORMED_STRUCTURED_OUTPUT,
            "structured_output.document",
            representation=representation,
            reason=reason,
        )

    def _too_large(self, representation: str, size: int, limit: int) -> TaskOutputFailure:
        return self._failure(
            TaskFailureCode.RESPONSE_TOO_LARGE,
            "structured_output.response_size",
            representation=representation,
            response_size=size,
            response_size_limit=limit,
        )

    def _schema_failure(
        self,
        representation: str,
        reason: str,
        index: int | None = None,
    ) -> TaskOutputFailure:
        details: dict[str, object] = {"representation": representation, "reason": reason}
        if index is not None:
            details["record_index"] = index
        return self._failure(
            TaskFailureCode.OUTPUT_SCHEMA_VIOLATION,
            "structured_output.schema",
            **details,
        )


__all__ = [
    "EnvelopeExtractor",
    "ParseResult",
    "ParsedTaskRecords",
    "StructuredOutputParser",
]
