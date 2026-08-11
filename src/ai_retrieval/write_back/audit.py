"""Append-oriented in-memory write-back audit sink for composition and tests."""

from threading import RLock

from ai_retrieval.write_back.models import WriteBackAuditEvent


class InMemoryWriteBackAuditSink:
    def __init__(self, fail: bool = False) -> None:
        self._events: list[WriteBackAuditEvent] = []
        self._fail = fail
        self._lock = RLock()

    def append(self, event: WriteBackAuditEvent) -> None:
        if self._fail:
            raise RuntimeError("write-back audit sink unavailable")
        with self._lock:
            self._events.append(event)

    @property
    def events(self) -> tuple[WriteBackAuditEvent, ...]:
        with self._lock:
            return tuple(self._events)
