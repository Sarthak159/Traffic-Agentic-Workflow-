"""Versioned domain contracts shared by ingestion, providers, workflows, alerts and evaluation.

Application code, never model output, supplies window IDs, times, statuses and record IDs.
Provider-specific payloads must be normalized into these models before use.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

FRAME_ID_PATTERN = r"^f-[0-9]+\.[0-9]{6}$"
SOURCE_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$"
WINDOW_ID_PATTERN = r"^w-[0-9a-f]{24}$"
LOCAL_REF_PATTERN = r"^[A-Za-z0-9_][A-Za-z0-9_./-]{0,199}$"

MAX_OBSERVATIONS = 24
MAX_LIMITATIONS = 12

ShortText = Annotated[str, StringConstraints(min_length=1, max_length=500, strip_whitespace=True)]
FlagText = Annotated[str, StringConstraints(min_length=1, max_length=120)]
FrameId = Annotated[str, StringConstraints(pattern=FRAME_ID_PATTERN)]
Uncertainty = Literal["low", "medium", "high", "unknown"]
RecordId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}$")]
FAILURE_STATUSES = ("failed", "timeout", "budget_exhausted")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class FrameRef(StrictModel):
    frame_id: FrameId
    source_time_s: float = Field(ge=0)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    # App-owned handle relative to a run's private media store; never an absolute path or URL.
    local_ref: str = Field(pattern=LOCAL_REF_PATTERN)

    @model_validator(mode="after")
    def _relative_handle(self) -> FrameRef:
        if ".." in self.local_ref.split("/"):
            raise ValueError("local_ref must not traverse parent directories")
        return self


class ObservationWindow(StrictModel):
    schema_version: Literal["1"] = "1"
    source_id: str = Field(pattern=SOURCE_ID_PATTERN)
    window_id: str = Field(pattern=WINDOW_ID_PATTERN)
    start_s: float = Field(ge=0)
    cutoff_s: float = Field(ge=0)
    frames: tuple[FrameRef, ...] = Field(min_length=1, max_length=256)
    quality_flags: tuple[FlagText, ...] = ()

    @model_validator(mode="after")
    def _check_interval(self) -> ObservationWindow:
        if self.start_s > self.cutoff_s:
            raise ValueError("start_s must not exceed cutoff_s")
        ids = [frame.frame_id for frame in self.frames]
        if len(set(ids)) != len(ids):
            raise ValueError("frame IDs must be unique within a window")
        times = [frame.source_time_s for frame in self.frames]
        if any(later <= earlier for earlier, later in zip(times, times[1:], strict=False)):
            raise ValueError("frame timestamps must be strictly increasing")
        if times[0] <= self.start_s or times[-1] > self.cutoff_s:
            raise ValueError("every frame must lie in the interval (start_s, cutoff_s]")
        return self

    @property
    def frame_ids(self) -> frozenset[str]:
        return frozenset(frame.frame_id for frame in self.frames)


class Observation(StrictModel):
    text: ShortText
    frame_ids: tuple[FrameId, ...] = Field(default=(), max_length=32)
    kind: Literal["observed", "inferred", "limitation"]

    @model_validator(mode="after")
    def _observed_needs_frames(self) -> Observation:
        if self.kind == "observed" and not self.frame_ids:
            raise ValueError("an observed claim must reference at least one frame")
        return self


class SpecialistEvidence(StrictModel):
    schema_version: Literal["1"] = "1"
    specialist_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")
    window_id: str = Field(pattern=WINDOW_ID_PATTERN)
    status: Literal["ok", "insufficient_evidence", "failed", "timeout", "budget_exhausted"]
    observations: tuple[Observation, ...] = Field(default=(), max_length=MAX_OBSERVATIONS)
    uncertainty: Uncertainty = "unknown"
    limitations: tuple[ShortText, ...] = Field(default=(), max_length=MAX_LIMITATIONS)

    @model_validator(mode="after")
    def _failure_is_not_evidence(self) -> SpecialistEvidence:
        if self.status in FAILURE_STATUSES:
            if self.observations:
                raise ValueError(f"a {self.status} specialist result cannot carry observations")
            if self.uncertainty != "unknown":
                raise ValueError(f"a {self.status} specialist result must have unknown uncertainty")
        return self


class Assessment(StrictModel):
    schema_version: Literal["1"] = "1"
    window_id: str = Field(pattern=WINDOW_ID_PATTERN)
    status: Literal["ok", "unknown", "failed", "timeout", "budget_exhausted"]
    risk_level: Literal["none", "low", "medium", "high", "unknown"]
    event_state: Literal[
        "no_supported_conflict",
        "potential_conflict",
        "suspected_incident",
        "observed_incident",
        "unknown",
    ]
    event_type: Literal["vehicle_pedestrian", "vehicle_vehicle", "other", "unknown"]
    horizon_s: Literal[2, 3, 5]
    observations: tuple[Observation, ...] = Field(default=(), max_length=MAX_OBSERVATIONS)
    specialist_result_ids: tuple[RecordId, ...] = Field(default=(), max_length=32)
    uncertainty: Uncertainty = "unknown"
    limitations: tuple[ShortText, ...] = Field(default=(), max_length=MAX_LIMITATIONS)

    @model_validator(mode="after")
    def _check_consistency(self) -> Assessment:
        if self.status != "ok" and (self.risk_level != "unknown" or self.event_state != "unknown"):
            raise ValueError("a non-ok assessment must have unknown risk_level and event_state")
        if self.status in FAILURE_STATUSES:
            # A failure is not a risk judgment and cannot carry model claims.
            if self.observations or self.event_type != "unknown" or self.uncertainty != "unknown":
                raise ValueError(
                    f"a {self.status} assessment cannot carry observations, "
                    "an event type or uncertainty"
                )
        check_judgment(self.risk_level, self.event_state, self.observations)
        return self


def check_judgment(risk_level: str, event_state: str, observations: Iterable[Observation]) -> None:
    """Cross-field rules shared by `Assessment` and the model-output schema; raises ValueError."""
    if event_state == "unknown" and risk_level != "unknown":
        raise ValueError("unknown event_state requires unknown risk_level")
    if event_state == "no_supported_conflict" and risk_level in ("medium", "high"):
        raise ValueError("no_supported_conflict is inconsistent with medium/high risk")
    needs_evidence = risk_level == "high" or event_state in (
        "suspected_incident",
        "observed_incident",
    )
    if needs_evidence and not any(obs.kind == "observed" for obs in observations):
        raise ValueError(
            "high-risk and incident claims need at least one observed, frame-referenced claim"
        )


def unknown_frame_refs(observations: Iterable[Observation], window: ObservationWindow) -> list[str]:
    """Return frame IDs referenced by observations that are not part of the supplied window.

    Referential validity does not prove that a description is visually correct.
    """
    allowed = window.frame_ids
    return sorted({fid for obs in observations for fid in obs.frame_ids if fid not in allowed})
