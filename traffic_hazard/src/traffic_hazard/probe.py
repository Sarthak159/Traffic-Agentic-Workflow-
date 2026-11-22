"""Model capability probe (TRAFFIC_AGENT_PIPELINE.md section 6, capability probe matrix).

Every attempt goes through `guarded_generate`, and every response through the shared strict parser
and output validation. Probes whose acceptance needs a model that actually inspects media or calls
tools are reported as `not_evaluable` for the fake provider. A fake result is harness evidence
only, never capability evidence.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

from .config import AppConfig
from .contracts import ObservationWindow
from .providers.base import ModelProvider, ModelRequest, ModelResult, OutputSchema
from .providers.guard import guarded_generate
from .providers.output import validate_output
from .video.reader import InMemoryVideoSource, PyAVVideoSource, RawFrame
from .video.synthetic import cfr_times, synthetic_image
from .video.windows import build_windows

PROBE_PROMPT_VERSION = "probe_v1"
PROBE_INSTRUCTIONS: dict[OutputSchema, str] = {
    "assessment": (
        "Capability probe. Describe only what is visible in the supplied ordered frames. "
        "Return one JSON object with risk_level, event_state, event_type, observations "
        "(text, frame_ids, kind), uncertainty and limitations. Do not identify people."
    ),
    "specialist_evidence": (
        "Capability probe. Report visible motion across the supplied ordered frames. Return one "
        "JSON object with sufficient, observations (text, frame_ids, kind), uncertainty and "
        "limitations."
    ),
}
MEDIA_PROBES = ("still_image", "ordered_frames", "reversed_frames", "control_frames")

ProbeStatus = Literal["passed", "failed", "not_evaluable", "skipped"]


@dataclass(frozen=True)
class ProbeOutcome:
    name: str
    status: ProbeStatus
    detail: str
    attempts: int = 0


def synthetic_probe_window(cfg: AppConfig, source_id: str = "probe-synthetic") -> ObservationWindow:
    """First causal window of an in-memory synthetic clip, sampled with the run's video config.

    The clip runs past the first eligible cutoff (the first interval multiple at or after one full
    window), so every valid video config yields a window.
    """
    first_cutoff = math.ceil(cfg.video.window / cfg.video.interval) * cfg.video.interval
    seconds = float(first_cutoff) + 1.0
    raw = [RawFrame(Fraction(str(t)), synthetic_image(t)) for t in cfr_times(seconds, 10)]
    source = InMemoryVideoSource(raw, origin=Fraction(0))
    result = next(iter(build_windows(source, cfg.video, source_id)), None)
    if result is None:
        raise RuntimeError("synthetic probe clip produced no window")
    return result.window


def first_video_window(video: Path, cfg: AppConfig, source_id: str) -> ObservationWindow | None:
    for result in build_windows(PyAVVideoSource(video), cfg.video, source_id):
        return result.window
    return None


def _request(cfg: AppConfig, window: ObservationWindow, schema: OutputSchema) -> ModelRequest:
    return ModelRequest(
        role="main" if schema == "assessment" else "motion",
        prompt_version=PROBE_PROMPT_VERSION,
        instructions=PROBE_INSTRUCTIONS[schema],
        window=window,
        output_schema=schema,
        max_output_tokens=cfg.model.max_output_tokens,
        timeout_s=cfg.model.request_timeout_seconds,
    )


def _percentile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile; None when there are no values."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def _json_probe(name: str, result: ModelResult, schema: OutputSchema, window) -> ProbeOutcome:
    if result.error is not None:
        return ProbeOutcome(name, "failed", f"{result.error.kind}: {result.error.message}", 1)
    assert result.parsed is not None
    validation = validate_output(schema, result.parsed, window)
    if not validation.ok:
        return ProbeOutcome(name, "failed", "; ".join(validation.errors), 1)
    return ProbeOutcome(name, "passed", f"valid {schema} output with in-window frame IDs", 1)


async def run_probe(
    cfg: AppConfig, provider: ModelProvider, window: ObservationWindow, repeats: int = 5
) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    simulated = cfg.model.provider == "fake"
    outcomes: list[ProbeOutcome] = []
    results: list[ModelResult] = []

    first = await guarded_generate(provider, _request(cfg, window, "assessment"))
    results.append(first)
    reached = first.error is None or first.error.kind == "malformed_output"
    if not reached:
        assert first.error is not None
        outcomes.append(ProbeOutcome("connectivity", "failed", first.error.kind, 1))
    elif first.model_id != cfg.model.model_id:
        outcomes.append(ProbeOutcome("connectivity", "failed", "effective model ID differs", 1))
    else:
        outcomes.append(ProbeOutcome("connectivity", "passed", "response received", 1))

    outcomes.append(_json_probe("json_assessment", first, "assessment", window))
    specialist = await guarded_generate(provider, _request(cfg, window, "specialist_evidence"))
    results.append(specialist)
    outcomes.append(_json_probe("json_specialist", specialist, "specialist_evidence", window))

    for name in MEDIA_PROBES:
        reason = (
            "fake provider does not inspect media"
            if simulated
            else "media probe evaluation is not implemented for this provider"
        )
        outcomes.append(ProbeOutcome(name, "not_evaluable", reason))
    outcomes.append(
        ProbeOutcome("tool_selection", "not_evaluable", "no tool-call interface yet (Phase 5)")
    )
    outcomes.append(
        ProbeOutcome("native_video", "skipped", f"transport is {cfg.video.transport!r}")
    )

    timings: list[float] = []
    failures: dict[str, int] = {}
    for _ in range(repeats):
        result = await guarded_generate(provider, _request(cfg, window, "assessment"))
        results.append(result)
        timings.append(result.elapsed_s)
        if result.error is not None:
            failures[result.error.kind] = failures.get(result.error.kind, 0) + 1
    usage_reported = sum(
        1 for r in results if r.usage.input_tokens is not None or r.usage.output_tokens is not None
    )
    latency = {
        "attempts": repeats,
        "p50_s": _percentile(timings, 0.5),
        "p95_s": _percentile(timings, 0.95),
        "failures": failures,
        "usage_reported_attempts": usage_reported,
        "request_payload_bytes": None,
    }
    outcomes.append(
        ProbeOutcome(
            "cost_latency",
            "failed" if failures else "passed",
            f"{repeats} attempts, failures={failures or 'none'}",
            repeats,
        )
    )

    return {
        "schema_version": "1",
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "simulated": simulated,
        "notice": (
            "Fake provider: verifies the probe harness only; not evidence of model capability."
            if simulated
            else "Real provider probe."
        ),
        "provider": cfg.model.provider,
        "configured_model_id": cfg.model.model_id,
        "effective_model_ids": sorted({r.model_id for r in results}),
        "transport": cfg.video.transport,
        "output_mode": "json_object_text",
        "retries": "none (allow_retries is not implemented; every attempt is counted)",
        "config_fingerprint": cfg.fingerprint(),
        "prompt_version": PROBE_PROMPT_VERSION,
        "window_id": window.window_id,
        "total_attempts": len(results),
        "probes": [asdict(o) for o in outcomes],
        "latency": latency,
    }
