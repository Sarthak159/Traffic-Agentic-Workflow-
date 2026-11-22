"""Model provider boundary: typed request/result shapes independent of any vendor payload."""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import Field, model_validator

from ..contracts import ObservationWindow, SpecialistEvidence, StrictModel

OutputSchema = Literal["assessment", "specialist_evidence"]
ErrorKind = Literal["timeout", "malformed_output", "provider_error", "cancelled"]


class ModelRequest(StrictModel):
    role: str = Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")
    prompt_version: str = Field(min_length=1, max_length=64)
    instructions: str = Field(min_length=1, max_length=20000)
    window: ObservationWindow
    scoped_evidence: tuple[SpecialistEvidence, ...] = Field(default=(), max_length=16)
    output_schema: OutputSchema
    max_output_tokens: int = Field(ge=1, le=32768)
    timeout_s: float = Field(gt=0)


class Usage(StrictModel):
    # None means the provider did not report the value; it is never coerced to zero.
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class ModelError(StrictModel):
    kind: ErrorKind
    # Sanitized message: no credentials, request bodies or media.
    message: str = Field(max_length=500)


class ModelResult(StrictModel):
    """Outcome of exactly one provider attempt.

    `parsed` is a JSON object that the caller must still validate; it is never trusted as-is.
    """

    parsed: dict[str, Any] | None = None
    raw_text: str | None = Field(default=None, max_length=100_000)
    response_id: str | None = None
    model_id: str
    usage: Usage = Usage()
    elapsed_s: float = Field(ge=0)
    error: ModelError | None = None

    @model_validator(mode="after")
    def _parsed_xor_error(self) -> ModelResult:
        if (self.parsed is None) == (self.error is None):
            raise ValueError(
                "a ModelResult carries either parsed output or an error, not both or neither"
            )
        return self

    @property
    def ok(self) -> bool:
        return self.error is None


@runtime_checkable
class ModelProvider(Protocol):
    """Async boundary every backend implements; one call is one counted model attempt."""

    model_id: str

    async def generate(self, request: ModelRequest) -> ModelResult: ...
