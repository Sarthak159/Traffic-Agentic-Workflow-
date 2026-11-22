"""Deterministic offline provider for tests, CI and scripted mode. Never contacts a network."""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from ..hashing import fingerprint
from .base import ModelError, ModelRequest, ModelResult
from .output import MAX_RAW_CHARS, MalformedOutput, parse_model_json

FakeKind = Literal["json", "raw", "timeout", "error", "raise"]


@dataclass(frozen=True)
class FakeResponse:
    """One scripted provider behavior.

    `json` serializes `payload` (or the built-in default for the requested schema). `raw` returns
    `raw_text` verbatim, which is how malformed output is scripted. Both go through the same strict
    parser a real provider uses. `timeout` simulates exceeding the request deadline, `error` a
    reported provider failure, and `raise` a provider that raises (for the guard).
    """

    kind: FakeKind = "json"
    payload: Mapping[str, Any] | None = None
    raw_text: str = "{not valid json"
    delay_s: float = 0.0
    message: str = "scripted provider failure"


@dataclass
class FakeProvider:
    """Scripted provider keyed by role; a sequence plays in order, then repeats its last entry."""

    script: Mapping[str, FakeResponse | Sequence[FakeResponse]] = field(default_factory=dict)
    model_id: str = "fake-v1"
    calls: list[ModelRequest] = field(default_factory=list, init=False)
    _per_role: dict[str, int] = field(
        default_factory=lambda: defaultdict(int), init=False, repr=False
    )

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def _next_response(self, role: str) -> FakeResponse:
        entry = self.script.get(role)
        index = self._per_role[role]
        self._per_role[role] += 1
        if entry is None:
            return FakeResponse()
        if isinstance(entry, FakeResponse):
            return entry
        if not entry:
            return FakeResponse()
        return entry[min(index, len(entry) - 1)]

    async def generate(self, request: ModelRequest) -> ModelResult:
        started = time.monotonic()
        self.calls.append(request)
        response = self._next_response(request.role)
        response_id = (
            "fake-"
            + fingerprint([request.model_dump(mode="json"), self._per_role[request.role]])[:16]
        )

        if response.kind == "timeout" or response.delay_s > request.timeout_s:
            await asyncio.sleep(min(response.delay_s, request.timeout_s, 0.05))
            return self._error(request, "timeout", "request exceeded its deadline", started)
        if response.delay_s:
            await asyncio.sleep(response.delay_s)
        if response.kind == "error":
            return self._error(request, "provider_error", response.message, started)
        if response.kind == "raise":
            raise RuntimeError(response.message)
        if response.kind == "raw":
            raw_text = response.raw_text
        else:
            payload = response.payload if response.payload is not None else default_payload(request)
            raw_text = json.dumps(payload, sort_keys=True)
        try:
            parsed = parse_model_json(raw_text)
        except MalformedOutput as exc:
            return ModelResult(
                raw_text=raw_text if len(raw_text) <= MAX_RAW_CHARS else None,
                response_id=response_id,
                model_id=self.model_id,
                elapsed_s=time.monotonic() - started,
                error=ModelError(kind="malformed_output", message=str(exc)),
            )
        return ModelResult(
            parsed=parsed,
            raw_text=raw_text,
            response_id=response_id,
            model_id=self.model_id,
            elapsed_s=time.monotonic() - started,
        )

    def _error(self, request: ModelRequest, kind: Any, message: str, started: float) -> ModelResult:
        return ModelResult(
            model_id=self.model_id,
            elapsed_s=time.monotonic() - started,
            error=ModelError(kind=kind, message=message),
        )


def default_payload(request: ModelRequest) -> dict[str, Any]:
    """Deterministic default output for a schema; references only frames in the request window."""
    last_frame = request.window.frames[-1].frame_id
    if request.output_schema == "assessment":
        return {
            "risk_level": "low",
            "event_state": "no_supported_conflict",
            "event_type": "vehicle_pedestrian",
            "observations": [
                {
                    "text": "Fake provider: no developing conflict visible.",
                    "frame_ids": [last_frame],
                    "kind": "observed",
                }
            ],
            "uncertainty": "medium",
            "limitations": ["Deterministic fake output; not a real visual assessment."],
        }
    return {
        # Execution status is app-owned, so model output never includes it.
        "sufficient": True,
        "observations": [
            {
                "text": f"Fake {request.role} evidence.",
                "frame_ids": [last_frame],
                "kind": "observed",
            }
        ],
        "uncertainty": "medium",
        "limitations": ["Deterministic fake output; not a real visual assessment."],
    }


def contradictory_assessment(frame_id: str) -> FakeResponse:
    """Valid JSON whose judgment is self-contradictory (high risk yet no supported conflict)."""
    return FakeResponse(
        payload={
            "risk_level": "high",
            "event_state": "no_supported_conflict",
            "event_type": "vehicle_pedestrian",
            "observations": [
                {"text": "Contradictory.", "frame_ids": [frame_id], "kind": "observed"}
            ],
            "uncertainty": "low",
        }
    )


def future_frame_assessment() -> FakeResponse:
    """Valid-looking output that cites a frame outside the request window."""
    return FakeResponse(
        payload={
            "risk_level": "low",
            "event_state": "no_supported_conflict",
            "event_type": "vehicle_pedestrian",
            "observations": [
                {"text": "Cites the future.", "frame_ids": ["f-9999.000000"], "kind": "observed"}
            ],
            "uncertainty": "low",
        }
    )


def malformed(raw_text: str = "{not valid json") -> FakeResponse:
    return FakeResponse(kind="raw", raw_text=raw_text)
