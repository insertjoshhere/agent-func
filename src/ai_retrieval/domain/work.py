"""Immutable query and model work descriptions."""

from dataclasses import dataclass, field

from ai_retrieval.domain.immutable import FrozenMapping


@dataclass(frozen=True)
class QueryWork:
    plan_id: str
    parameters: FrozenMapping = field(default_factory=lambda: FrozenMapping(()))

    def __post_init__(self) -> None:
        if not self.plan_id.strip():
            raise ValueError("query plan_id must not be blank")


@dataclass(frozen=True)
class ModelWork:
    task_type: str
    payload_reference: str
    input_ids: tuple[str, ...]
    required_capabilities: frozenset[str] = frozenset()
    task_definition_version: str | None = None
    estimated_input_tokens: int | None = None
    estimated_output_tokens: int | None = None

    def __post_init__(self) -> None:
        if not self.task_type.strip() or not self.payload_reference.strip():
            raise ValueError("model task_type and payload_reference must not be blank")
        if any(not value.strip() for value in self.input_ids):
            raise ValueError("model input identifiers must not be blank")
        if self.task_definition_version is not None and not self.task_definition_version.strip():
            raise ValueError("task definition version must not be blank")
        estimates = (self.estimated_input_tokens, self.estimated_output_tokens)
        if any(
            value is not None
            and (isinstance(value, bool) or not isinstance(value, int) or value < 0)
            for value in estimates
        ):
            raise ValueError("model token estimates must be nonnegative integers")


@dataclass(frozen=True)
class InteractiveTaskWork:
    """Task selection whose rows are populated only from the normalized read."""

    function: str
    parameters: FrozenMapping
    requested_definition_version: str | None = None

    def __post_init__(self) -> None:
        if not self.function.strip():
            raise ValueError("interactive task function must not be blank")
        if not isinstance(self.parameters, FrozenMapping):
            raise TypeError("interactive task parameters must be a FrozenMapping")
        if self.requested_definition_version is not None and not self.requested_definition_version.strip():
            raise ValueError("requested task definition version must not be blank")


@dataclass(frozen=True)
class InteractiveRequest:
    request_id: str
    query: QueryWork
    model_work: tuple[ModelWork, ...] = ()
    task: InteractiveTaskWork | None = None

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("request_id must not be blank")
        if self.task is not None and self.model_work:
            raise ValueError("interactive request cannot mix task work and legacy ModelWork")


@dataclass(frozen=True)
class BulkTaskWork:
    """Task selection whose durable rows are derived from bulk work items."""

    function: str
    parameters: FrozenMapping
    source_rows: FrozenMapping = field(default_factory=lambda: FrozenMapping(()))
    requested_definition_version: str | None = None

    def __post_init__(self) -> None:
        if not self.function.strip():
            raise ValueError("bulk task function must not be blank")
        if not isinstance(self.parameters, FrozenMapping):
            raise TypeError("bulk task parameters must be a FrozenMapping")
        if not isinstance(self.source_rows, FrozenMapping):
            raise TypeError("bulk task source_rows must be a FrozenMapping")
        if self.requested_definition_version is not None and not self.requested_definition_version.strip():
            raise ValueError("requested task definition version must not be blank")


@dataclass(frozen=True)
class BatchJob:
    job_id: str
    item_ids: tuple[str, ...]
    model_work: tuple[ModelWork, ...] = ()
    task: BulkTaskWork | None = None

    def __post_init__(self) -> None:
        if not self.job_id.strip():
            raise ValueError("job_id must not be blank")
        if not self.item_ids or any(not value.strip() for value in self.item_ids):
            raise ValueError("a batch job requires non-blank item identifiers")
        if self.task is not None and self.model_work:
            raise ValueError("batch job cannot mix task work and legacy ModelWork")
