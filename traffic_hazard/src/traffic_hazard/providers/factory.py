"""Build the configured provider. Real backends are added here in the Phase 2 real-model work."""

from __future__ import annotations

from ..config import ModelSettings
from .base import ModelProvider
from .fake import FakeProvider


def create_provider(settings: ModelSettings) -> ModelProvider:
    if settings.provider == "fake":
        return FakeProvider(model_id=settings.model_id)
    raise ValueError(f"unsupported provider {settings.provider!r}")
