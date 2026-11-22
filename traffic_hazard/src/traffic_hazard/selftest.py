"""Offline self-test: config rejection, contract validation and the deterministic fake provider."""

from __future__ import annotations

from collections.abc import Callable

from .config import AppConfig, ConfigError, validate_config
from .contracts import Assessment, FrameRef, ObservationWindow, SpecialistEvidence
from .providers.base import ModelRequest
from .providers.fake import FakeProvider, FakeResponse


def synthetic_window() -> ObservationWindow:
    """A tiny, valid window built in memory; no media is decoded."""
    frames = tuple(
        FrameRef(
            frame_id=f"f-{t:.6f}",
            source_time_s=t,
            content_sha256=f"{i:064x}",
            local_ref=f"media/{i:064x}.jpg",
        )
        for i, t in enumerate((4.25, 4.5, 4.75, 5.0), start=1)
    )
    return ObservationWindow(
        source_id="selftest", window_id="w-" + "0" * 24, start_s=0.0, cutoff_s=5.0, frames=frames
    )


def _request(cfg: AppConfig, window: ObservationWindow, role: str, schema: str) -> ModelRequest:
    return ModelRequest(
        role=role,
        prompt_version="selftest_v1",
        instructions="Self-test request.",
        window=window,
        output_schema=schema,  # type: ignore[arg-type]
        max_output_tokens=cfg.model.max_output_tokens,
        timeout_s=cfg.model.request_timeout_seconds,
    )


async def run_self_test(cfg: AppConfig) -> bool:
    from .cli import sdk_tracing_is_noop

    results: list[tuple[str, bool]] = []

    def check(name: str, fn: Callable[[], bool]) -> None:
        try:
            ok = bool(fn())
        except Exception as exc:
            print(f"  error in {name}: {type(exc).__name__}: {exc}")
            ok = False
        results.append((name, ok))

    window = synthetic_window()
    base = cfg.model_dump(mode="json")

    def rejects(mutate: Callable[[dict], None]) -> bool:
        data = cfg.model_dump(mode="json")
        mutate(data)
        try:
            validate_config(data)
        except ConfigError:
            return True
        return False

    check("config accepts loaded file", lambda: validate_config(base) == cfg)
    check("config rejects unknown key", lambda: rejects(lambda d: d["video"].update(fps=4)))
    check(
        "config rejects fixed role outside pool",
        lambda: rejects(lambda d: d["workflow"].update(fixed_specialists=["temporal"])),
    )
    check(
        "config rejects frame cap overflow",
        lambda: rejects(lambda d: d["video"].update(sample_fps=10.0)),
    )
    check("sdk remote tracing disabled", sdk_tracing_is_noop)

    provider = FakeProvider()
    first = await provider.generate(_request(cfg, window, "main", "assessment"))
    second = await FakeProvider().generate(_request(cfg, window, "main", "assessment"))
    check(
        "fake provider deterministic",
        lambda: first.parsed == second.parsed and first.response_id == second.response_id,
    )
    check(
        "fake assessment validates as typed Assessment",
        lambda: (
            Assessment(
                window_id=window.window_id,
                status="ok",
                horizon_s=cfg.workflow.horizon_seconds,
                **first.parsed,
            )
            is not None
        ),
    )
    spec = await provider.generate(_request(cfg, window, "motion", "specialist_evidence"))
    check(
        "fake specialist validates as SpecialistEvidence",
        lambda: (
            SpecialistEvidence(
                specialist_id="motion", window_id=window.window_id, status="ok", **spec.parsed
            )
            is not None
        ),
    )
    check("fake call counter", lambda: provider.call_count == 2)

    failing = FakeProvider(
        script={
            "main": [
                FakeResponse(kind="malformed"),
                FakeResponse(kind="timeout"),
                FakeResponse(kind="error"),
            ]
        }
    )
    kinds = []
    for _ in range(3):
        result = await failing.generate(_request(cfg, window, "main", "assessment"))
        kinds.append(result.error.kind if result.error else None)
    check(
        "fake failure modes are typed errors",
        lambda: kinds == ["malformed_output", "timeout", "provider_error"],
    )

    for name, ok in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    passed = all(ok for _, ok in results)
    print(
        f"self-test {'passed' if passed else 'FAILED'}: "
        f"{sum(ok for _, ok in results)}/{len(results)}"
    )
    return passed
