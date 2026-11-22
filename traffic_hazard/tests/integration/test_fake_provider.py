from __future__ import annotations

import pytest
from pydantic import ValidationError
from traffic_hazard.providers.base import ModelError, ModelProvider, ModelRequest, ModelResult
from traffic_hazard.providers.fake import FakeProvider, FakeResponse
from traffic_hazard.selftest import synthetic_window


def _request(
    role: str = "main", schema: str = "assessment", timeout_s: float = 1.0
) -> ModelRequest:
    return ModelRequest(
        role=role,
        prompt_version="v1",
        instructions="x",
        window=synthetic_window(),
        output_schema=schema,
        max_output_tokens=100,
        timeout_s=timeout_s,
    )


async def test_deterministic_typed_result() -> None:
    a = await FakeProvider().generate(_request())
    b = await FakeProvider().generate(_request())
    assert isinstance(FakeProvider(), ModelProvider)
    assert a.ok and a.parsed == b.parsed and a.response_id == b.response_id
    assert (
        a.usage.input_tokens is None and a.usage.output_tokens is None
    )  # Unknown usage is null, not zero.
    assert a.parsed["observations"][0]["frame_ids"] == ["f-5.000000"]


async def test_scripted_sequence_and_failures() -> None:
    provider = FakeProvider(
        script={
            "main": [
                FakeResponse(payload={"risk_level": "none"}),
                FakeResponse(kind="malformed"),
                FakeResponse(kind="error", message="boom"),
            ]
        }
    )
    first = await provider.generate(_request())
    malformed = await provider.generate(_request())
    error = await provider.generate(_request())
    repeat = await provider.generate(_request())
    assert first.parsed == {"risk_level": "none"}
    assert (
        malformed.parsed is None
        and malformed.raw_text
        and malformed.error.kind == "malformed_output"
    )
    assert error.error == ModelError(kind="provider_error", message="boom")
    assert repeat.error.kind == "provider_error"  # The last scripted entry repeats.
    assert provider.call_count == 4


async def test_timeout_from_delay_exceeding_deadline() -> None:
    provider = FakeProvider(script={"main": FakeResponse(delay_s=5.0)})
    result = await provider.generate(_request(timeout_s=0.01))
    assert result.error.kind == "timeout" and result.elapsed_s < 1.0
    delayed = await FakeProvider(script={"main": FakeResponse(delay_s=0.01)}).generate(_request())
    assert delayed.ok and delayed.elapsed_s >= 0.01


async def test_scripted_payload_is_not_shared_with_callers() -> None:
    provider = FakeProvider(
        script={"main": FakeResponse(payload={"observations": [{"text": "a"}]})}
    )
    first = await provider.generate(_request())
    first.parsed["observations"][0]["text"] = "mutated"
    second = await provider.generate(_request())
    assert second.parsed == {"observations": [{"text": "a"}]}


async def test_specialist_default_payload_and_roles() -> None:
    provider = FakeProvider()
    result = await provider.generate(_request(role="motion", schema="specialist_evidence"))
    assert "status" not in result.parsed  # Execution status is app-owned.
    with pytest.raises(ValidationError):
        _request(role="../etc")


def test_result_requires_exactly_one_outcome() -> None:
    with pytest.raises(ValidationError):
        ModelResult(model_id="m", elapsed_s=0)
    with pytest.raises(ValidationError):
        ModelResult(
            model_id="m", elapsed_s=0, parsed={}, error=ModelError(kind="timeout", message="t")
        )
