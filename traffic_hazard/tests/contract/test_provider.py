"""Contract every ModelProvider must satisfy, plus the shared strict output path."""

from __future__ import annotations

import asyncio

import pytest
from traffic_hazard.config import load_config
from traffic_hazard.providers.base import ModelProvider, ModelRequest, ModelResult
from traffic_hazard.providers.factory import create_provider
from traffic_hazard.providers.fake import (
    FakeProvider,
    FakeResponse,
    contradictory_assessment,
    future_frame_assessment,
    malformed,
)
from traffic_hazard.providers.guard import guarded_generate
from traffic_hazard.providers.output import MalformedOutput, parse_model_json, validate_output
from traffic_hazard.selftest import synthetic_window

from tests.conftest import CONFIGS

WINDOW = synthetic_window()


def _request(
    role: str = "main", schema: str = "assessment", timeout_s: float = 1.0
) -> ModelRequest:
    return ModelRequest(
        role=role,
        prompt_version="v1",
        instructions="x",
        window=WINDOW,
        output_schema=schema,
        max_output_tokens=100,
        timeout_s=timeout_s,
    )


def _providers() -> list[ModelProvider]:
    """Every provider constructible from shipped configs; real backends join this list later."""
    return [create_provider(load_config(CONFIGS / "local_fake.yaml").model)]


@pytest.mark.parametrize("provider", _providers(), ids=lambda p: type(p).__name__)
@pytest.mark.parametrize("schema", ["assessment", "specialist_evidence"])
async def test_provider_returns_validated_typed_output(
    provider: ModelProvider, schema: str
) -> None:
    result = await guarded_generate(provider, _request(schema=schema))
    assert isinstance(result, ModelResult) and result.ok
    assert result.model_id == provider.model_id
    assert validate_output(schema, result.parsed, WINDOW).ok
    for app_owned in ("window_id", "status", "specialist_id", "schema_version"):
        assert app_owned not in result.parsed


def test_factory_uses_configured_model_id() -> None:
    cfg = load_config(CONFIGS / "local_fake.yaml")
    assert (
        create_provider(cfg.model.model_copy(update={"model_id": "fake-v2"})).model_id == "fake-v2"
    )


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "[]",
        '"text"',
        "```json\n{}\n```",
        'Sure! {"risk_level": "low"}',
        '{"a": 1} {"b": 2}',
        '{"a": 1, "a": 2}',
        '{"a": NaN}',
        '{"a": Infinity}',
        '{"a": 1e999}',
        "[" * 50_000,
        '{"a": ' + "9" * 5000 + "}",
        "x" * 100_001,
    ],
)
def test_parser_rejects_anything_but_one_json_object(raw: str) -> None:
    with pytest.raises(MalformedOutput):
        parse_model_json(raw)


def test_parser_accepts_one_object_with_outer_whitespace() -> None:
    assert parse_model_json('  \n{"a": [1, {"b": null}]}\n') == {"a": [1, {"b": None}]}


def test_output_validation_rejects_app_owned_fields_contradictions_and_foreign_frames() -> None:
    good = {
        "risk_level": "low",
        "event_state": "no_supported_conflict",
        "event_type": "vehicle_pedestrian",
        "observations": [{"text": "ok", "frame_ids": ["f-5.000000"], "kind": "observed"}],
        "uncertainty": "medium",
    }
    assert validate_output("assessment", good, WINDOW).ok
    assert not validate_output("assessment", {**good, "status": "ok"}, WINDOW).ok
    assert not validate_output("assessment", {**good, "window_id": "w-" + "0" * 24}, WINDOW).ok
    contradictory = {**good, "risk_level": "high", "uncertainty": "low"}
    errors = validate_output("assessment", contradictory, WINDOW).errors
    assert errors and "inconsistent" in errors[0]
    foreign = {
        **good,
        "observations": [{"text": "x", "frame_ids": ["f-6.000000"], "kind": "observed"}],
    }
    assert validate_output("assessment", foreign, WINDOW).errors == (
        "observations reference 1 unknown frame IDs",
    )
    secret = {**good, "risk_level": "sk-secret-value"}
    assert "sk-secret-value" not in " ".join(validate_output("assessment", secret, WINDOW).errors)
    assert not validate_output("specialist_evidence", {"uncertainty": "low"}, WINDOW).ok
    assert validate_output(
        "specialist_evidence", {"sufficient": False, "uncertainty": "high"}, WINDOW
    ).ok


async def test_fake_failure_fixtures_flow_through_the_shared_path() -> None:
    provider = FakeProvider(
        script={
            "main": [
                malformed("```json\n{}\n```"),
                contradictory_assessment("f-5.000000"),
                future_frame_assessment(),
                FakeResponse(kind="timeout"),
            ]
        }
    )
    fenced = await guarded_generate(provider, _request())
    contradictory = await guarded_generate(provider, _request())
    future = await guarded_generate(provider, _request())
    timeout = await guarded_generate(provider, _request())
    assert fenced.error.kind == "malformed_output" and fenced.raw_text.startswith("```")
    assert contradictory.ok and not validate_output("assessment", contradictory.parsed, WINDOW).ok
    assert future.ok and not validate_output("assessment", future.parsed, WINDOW).ok
    assert timeout.error.kind == "timeout"


class _Raising:
    model_id = "raising"

    async def generate(self, request: ModelRequest) -> ModelResult:
        raise RuntimeError("leaked sk-secret in message")


class _Hanging:
    model_id = "hanging"

    def __init__(self) -> None:
        self.cancelled = False

    async def generate(self, request: ModelRequest) -> ModelResult:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        raise AssertionError("unreachable")


class _WrongType:
    model_id = "wrong"

    async def generate(self, request: ModelRequest):  # type: ignore[no-untyped-def]
        return {"risk_level": "high"}


async def test_guard_turns_exceptions_into_sanitized_typed_errors() -> None:
    result = await guarded_generate(_Raising(), _request())
    assert result.error.kind == "provider_error" and result.parsed is None
    assert result.error.message == "provider raised RuntimeError"
    assert result.model_id == "raising"
    wrong = await guarded_generate(_WrongType(), _request())
    assert wrong.error.kind == "provider_error"


async def test_guard_enforces_the_deadline_and_cancels_the_attempt() -> None:
    hanging = _Hanging()
    result = await guarded_generate(hanging, _request(timeout_s=0.05))
    assert result.error.kind == "timeout" and result.elapsed_s < 2
    assert hanging.cancelled


async def test_guard_propagates_caller_cancellation() -> None:
    task = asyncio.create_task(guarded_generate(_Hanging(), _request(timeout_s=30)))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_guard_with_raising_fake() -> None:
    provider = FakeProvider(script={"main": FakeResponse(kind="raise", message="boom")})
    assert (await guarded_generate(provider, _request())).error.kind == "provider_error"


async def test_hostile_raw_output_is_malformed_end_to_end_not_provider_error() -> None:
    hostile = [
        malformed('{"risk_level": "' + "x" * 100_001 + '"}'),
        malformed('{"' + "k" * 600 + '": 1, "' + "k" * 600 + '": 2}'),
        malformed("[" * 50_000),
        malformed('{"a": ' + "9" * 5000 + "}"),
    ]
    provider = FakeProvider(script={"main": hostile})
    for _ in hostile:
        result = await guarded_generate(provider, _request())
        assert result.error.kind == "malformed_output", result.error
        assert len(result.error.message) < 100


def test_model_controlled_keys_never_reach_error_messages() -> None:
    good = {
        "risk_level": "low",
        "event_state": "no_supported_conflict",
        "event_type": "vehicle_pedestrian",
        "uncertainty": "medium",
        "observations": [{"text": "ok", "frame_ids": ["f-5.000000"], "kind": "observed"}],
    }
    secret = "IGNORE PREVIOUS sk-leaked-key"
    leaky = {**good, secret: 1}
    leaky["observations"] = [{**good["observations"][0], secret + "2": 1}]
    errors = validate_output("assessment", leaky, WINDOW).errors
    assert errors and all("sk-leaked" not in e and "IGNORE" not in e for e in errors)
    assert "<extra>" in errors[0]
    many = {**good, **{f"{secret}{i}": i for i in range(50)}}
    assert len(validate_output("assessment", many, WINDOW).errors) == 8
    with pytest.raises(MalformedOutput) as exc:
        parse_model_json('{"' + secret + '": 1, "' + secret + '": 2}')
    assert "sk-leaked" not in str(exc.value)


class _LateAnswer:
    """Swallows the guard's cancellation and answers anyway (a misbehaving backend)."""

    model_id = "late"

    async def generate(self, request: ModelRequest) -> ModelResult:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            pass
        return ModelResult(model_id="late", elapsed_s=0.0, parsed={"risk_level": "high"})


async def test_guard_rejects_late_results_and_measures_latency_itself() -> None:
    result = await guarded_generate(_LateAnswer(), _request(timeout_s=0.05))
    assert result.error is not None and result.error.kind == "timeout" and result.parsed is None
    slow = FakeProvider(script={"main": FakeResponse(delay_s=0.05)})
    timed = await guarded_generate(slow, _request())
    assert timed.ok and timed.elapsed_s >= 0.05
