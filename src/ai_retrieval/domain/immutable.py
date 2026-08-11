"""Small recursive immutable containers for configuration and parameters."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias


FrozenValue: TypeAlias = "str | int | float | bool | None | tuple[FrozenValue, ...] | frozenset[FrozenValue] | FrozenMapping"


@dataclass(frozen=True)
class FrozenMapping(Mapping[str, FrozenValue]):
    """An immutable, deterministic string-keyed mapping."""

    _items: tuple[tuple[str, FrozenValue], ...]

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, key: str) -> FrozenValue:
        for item_key, value in self._items:
            if item_key == key:
                return value
        raise KeyError(key)


def freeze(value: Any) -> FrozenValue:
    """Copy supported input into a recursively immutable representation."""
    if isinstance(value, FrozenMapping):
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("immutable mappings require string keys")
        return FrozenMapping(tuple(sorted((key, freeze(item)) for key, item in value.items())))
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(freeze(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported immutable value: {type(value).__name__}")
