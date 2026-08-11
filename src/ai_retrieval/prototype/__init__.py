"""Replaceable local adapters for the minimal executable prototype."""

from ai_retrieval.prototype.adapters import (
    AllowReadPolicy, FakeEffectAdapter, FakeModelProvider,
    FakeReadRelationalAdapter, FakeServiceAuthenticator, RecordingTelemetry,
)

__all__ = [
    "AllowReadPolicy", "FakeEffectAdapter", "FakeModelProvider",
    "FakeReadRelationalAdapter", "FakeServiceAuthenticator", "RecordingTelemetry",
]
