"""Strict parsing and validation of model output, shared by every provider.

A model may propose judgments, observations and frame references. It never supplies app-owned
fields (window ID, execution status, times, record IDs). There is no text repair: the raw response
must be exactly one JSON object, or the attempt is a `malformed_output` failure.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import Field, ValidationError, model_validator

from ..contracts import (
    MAX_LIMITATIONS,
    MAX_OBSERVATIONS,
    Observation,
    ObservationWindow,
    ShortText,
    StrictModel,
    Uncertainty,
    check_judgment,
    unknown_frame_refs,
)
from .base import OutputSchema

MAX_RAW_CHARS = 100_000
MAX_REPORTED_ERRORS = 8


class AssessmentOutput(StrictModel):
    """What a model may propose for a main assessment (output schema `assessment`)."""

    risk_level: Literal["none", "low", "medium", "high", "unknown"]
    event_state: Literal[
        "no_supported_conflict",
        "potential_conflict",
        "suspected_incident",
        "observed_incident",
        "unknown",
    ]
    event_type: Literal["vehicle_pedestrian", "vehicle_vehicle", "other", "unknown"]
    observations: tuple[Observation, ...] = Field(default=(), max_length=MAX_OBSERVATIONS)
    uncertainty: Uncertainty
    limitations: tuple[ShortText, ...] = Field(default=(), max_length=MAX_LIMITATIONS)

    @model_validator(mode="after")
    def _consistent(self) -> AssessmentOutput:
        check_judgment(self.risk_level, self.event_state, self.observations)
        return self


class SpecialistOutput(StrictModel):
    """What a model may propose as specialist evidence (output schema `specialist_evidence`).

    `sufficient: false` lets the model say the footage cannot answer its question; the application
    maps that to the `insufficient_evidence` status. Execution statuses stay app-owned.
    """

    sufficient: bool
    observations: tuple[Observation, ...] = Field(default=(), max_length=MAX_OBSERVATIONS)
    uncertainty: Uncertainty
    limitations: tuple[ShortText, ...] = Field(default=(), max_length=MAX_LIMITATIONS)


OUTPUT_MODELS: dict[str, type[AssessmentOutput] | type[SpecialistOutput]] = {
    "assessment": AssessmentOutput,
    "specialist_evidence": SpecialistOutput,
}


class MalformedOutput(ValueError):
    """The raw response is not exactly one JSON object."""


def _reject_constant(name: str) -> Any:
    raise MalformedOutput(f"non-finite number {name} is not allowed")


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise MalformedOutput("non-finite number is not allowed")
    return value


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            # The key is model-controlled text, so it is not echoed.
            raise MalformedOutput("duplicate key in JSON object")
        result[key] = value
    return result


def parse_model_json(raw_text: str) -> dict[str, Any]:
    """Parse a raw response that must be exactly one JSON object (outer whitespace allowed)."""
    if len(raw_text) > MAX_RAW_CHARS:
        raise MalformedOutput("response exceeds the size limit")
    try:
        value = json.loads(
            raw_text,
            object_pairs_hook=_no_duplicates,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except MalformedOutput:
        raise
    except (ValueError, RecursionError):
        # Covers JSONDecodeError, too-deep nesting and integers beyond Python's digit limit.
        raise MalformedOutput("response is not valid JSON") from None
    if not isinstance(value, dict):
        raise MalformedOutput("response is not a JSON object")
    return value


@dataclass(frozen=True)
class OutputValidation:
    """Result of validating one parsed response; `output` is None when `errors` is non-empty."""

    output: AssessmentOutput | SpecialistOutput | None
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.output is not None


def validate_output(
    schema: OutputSchema, parsed: dict[str, Any], window: ObservationWindow
) -> OutputValidation:
    """Validate parsed model output for `schema` and check frame references against `window`.

    Error messages name schema fields and rules only. Pydantic keeps offending values out of `msg`,
    and unexpected keys (model-controlled names) are replaced by `<extra>`. At most
    `MAX_REPORTED_ERRORS` errors are reported.
    """
    model = OUTPUT_MODELS[schema]
    try:
        output = model.model_validate(parsed)
    except ValidationError as exc:
        errors = []
        for err in exc.errors()[:MAX_REPORTED_ERRORS]:
            loc = [str(part) for part in err["loc"]]
            if err["type"] == "extra_forbidden" and loc:
                loc[-1] = "<extra>"
            errors.append(f"{'.'.join(loc) or '<root>'}: {err['msg']}")
        return OutputValidation(None, tuple(errors))
    unknown = unknown_frame_refs(output.observations, window)
    if unknown:
        return OutputValidation(None, (f"observations reference {len(unknown)} unknown frame IDs",))
    return OutputValidation(output, ())
