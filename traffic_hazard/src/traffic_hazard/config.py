"""Application configuration: strict schema, YAML `extends` loading, cross-field validation.

`configs/defaults.yaml` is the single source of truth for default values. The models below declare
no defaults, so a missing key fails validation.
"""

from __future__ import annotations

import math
import re
from collections.abc import Hashable, Mapping
from fractions import Fraction
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    model_validator,
)

from .hashing import fingerprint

ROLE_ID_PATTERN = r"^[a-z][a-z0-9_]{0,31}$"
_MAX_EXTENDS_DEPTH = 5


class ConfigError(ValueError):
    """Raised when a configuration file cannot be loaded or is invalid."""


def _exact(expected: type):  # type: ignore[no-untyped-def]
    """Reject values whose type is not exactly `expected`, so `5.0` or `true` is not `5` or `1`."""

    def check(value: Any) -> Any:
        if type(value) is not expected:
            raise ValueError(f"expected {expected.__name__}, got {type(value).__name__}")
        return value

    return check


class _ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


# Strict floats reject `true` and quoted numbers while still accepting plain integers.
PositiveFloat = Annotated[float, Field(gt=0, strict=True)]


class VideoConfig(_ConfigModel):
    window_seconds: PositiveFloat
    assessment_interval_seconds: PositiveFloat
    sample_fps: PositiveFloat
    max_frames: StrictInt = Field(ge=1, le=256)
    max_width: StrictInt = Field(ge=16, le=7680)
    transport: Literal["frames", "native_video"]
    startup: Literal["require_full_window"]
    # A causal streaming builder cannot accept a frame earlier than one already seen.
    reject_out_of_order: Annotated[Literal[True], BeforeValidator(_exact(bool))]

    @property
    def window(self) -> Fraction:
        return Fraction(str(self.window_seconds))

    @property
    def interval(self) -> Fraction:
        return Fraction(str(self.assessment_interval_seconds))

    @property
    def tick_spacing(self) -> Fraction:
        return 1 / Fraction(str(self.sample_fps))

    @property
    def ticks_per_window(self) -> int:
        """Number of sampling ticks strictly inside `(t - window, t]`."""
        return math.ceil(self.window / self.tick_spacing)

    @model_validator(mode="after")
    def _check_frame_cap(self) -> VideoConfig:
        if self.ticks_per_window > self.max_frames:
            raise ValueError(
                f"sampling {self.sample_fps} FPS over {self.window_seconds} s needs "
                f"{self.ticks_per_window} frames, exceeding max_frames={self.max_frames}"
            )
        return self


class ModelSettings(_ConfigModel):
    # Phase 2 widens this to real providers; until then only the offline fake is implemented.
    provider: Literal["fake"]
    model_id: StrictStr = Field(min_length=1, max_length=200)
    revision: StrictStr | None
    temperature: float = Field(ge=0, le=2, strict=True)
    max_output_tokens: StrictInt = Field(ge=1, le=32768)
    request_timeout_seconds: PositiveFloat
    allow_retries: StrictBool


class WorkflowConfig(_ConfigModel):
    mode: Literal["single", "fixed", "adaptive", "scripted"]
    main_prompt: StrictStr = Field(pattern=r"^[a-z0-9_]+$")
    horizon_seconds: Annotated[Literal[2, 3, 5], BeforeValidator(_exact(int))]
    available_specialists: tuple[StrictStr, ...]
    fixed_specialists: tuple[StrictStr, ...]
    max_unique_specialists: StrictInt = Field(ge=0)
    max_specialist_calls: StrictInt = Field(ge=0)
    max_total_model_calls: StrictInt = Field(ge=1)
    max_total_input_tokens: StrictInt | None = Field(ge=1)
    max_elapsed_seconds: PositiveFloat
    max_parallel_specialists: StrictInt = Field(ge=1)
    allow_repeat_specialist: StrictBool

    @model_validator(mode="after")
    def _check_roles_and_budget(self) -> WorkflowConfig:
        for field in ("available_specialists", "fixed_specialists"):
            roles: tuple[str, ...] = getattr(self, field)
            bad = [role for role in roles if not re.fullmatch(ROLE_ID_PATTERN, role)]
            if bad:
                raise ValueError(f"{field} contains invalid role IDs: {bad}")
            if len(set(roles)) != len(roles):
                raise ValueError(f"{field} contains duplicate role IDs")
        outside = sorted(set(self.fixed_specialists) - set(self.available_specialists))
        if outside:
            raise ValueError(f"fixed_specialists {outside} are not in available_specialists")
        if self.max_unique_specialists > len(self.available_specialists):
            raise ValueError("max_unique_specialists exceeds the number of available_specialists")
        if self.max_specialist_calls > self.max_total_model_calls:
            raise ValueError("max_specialist_calls exceeds max_total_model_calls")
        if self.mode == "fixed":
            needed = len(self.fixed_specialists)
            if needed == 0:
                raise ValueError("fixed mode requires at least one fixed_specialists entry")
            if self.max_unique_specialists < needed or self.max_specialist_calls < needed:
                raise ValueError(
                    "fixed schedule exceeds max_unique_specialists or max_specialist_calls"
                )
            if self.max_total_model_calls < needed + 1:
                raise ValueError(
                    f"fixed schedule of {needed} specialists plus one synthesis call needs "
                    f"max_total_model_calls >= {needed + 1}"
                )
        if self.mode in ("adaptive", "scripted") and self.max_specialist_calls > 0:
            # Every specialist call needs the route call and the reserved synthesis call budgeted
            # too.
            needed = self.max_specialist_calls + 2
            if self.max_total_model_calls < needed:
                raise ValueError(
                    f"adaptive routing with {self.max_specialist_calls} specialist calls needs "
                    f"max_total_model_calls >= {needed} (route + specialists + synthesis)"
                )
        return self


class AlertsConfig(_ConfigModel):
    # Only local simulation exists; there is deliberately no value that selects real delivery.
    mode: Literal["simulated"]
    minimum_risk: Literal["low", "medium", "high"]
    require_evidence: StrictBool
    require_human_review: Annotated[Literal[True], BeforeValidator(_exact(bool))]
    max_staleness_seconds: PositiveFloat
    episode_gap_seconds: PositiveFloat
    cooldown_seconds: float = Field(ge=0, strict=True)
    consecutive_windows: StrictInt = Field(ge=1)


class LoggingConfig(_ConfigModel):
    save_raw_responses: StrictBool
    save_private_frames: StrictBool
    sdk_remote_tracing: StrictBool
    cache_mode: Literal["off", "read_only", "read_write"]


class AppConfig(_ConfigModel):
    schema_version: Annotated[Literal[1], BeforeValidator(_exact(int))]
    video: VideoConfig
    model: ModelSettings
    workflow: WorkflowConfig
    alerts: AlertsConfig
    logging: LoggingConfig

    def fingerprint(self) -> str:
        return fingerprint(self.model_dump(mode="json"))


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader without YAML 1.1 surprises.

    Only `true`/`false` are booleans (YAML 1.1 reads `off` as False), integers and floats are plain
    decimal (no base-60 `1:30` or octal `010`), and a repeated mapping key is an error instead of
    last-wins.
    """

    def construct_mapping(self, node, deep=False):  # type: ignore[no-untyped-def]
        seen: set[Any] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=True)
            if not isinstance(key, Hashable):
                raise yaml.constructor.ConstructorError(
                    None, None, "mapping keys must be scalars", key_node.start_mark
                )
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    None, None, f"duplicate key {key!r}", key_node.start_mark
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


_REPLACED_TAGS = {"tag:yaml.org,2002:bool", "tag:yaml.org,2002:int", "tag:yaml.org,2002:float"}
_StrictLoader.yaml_implicit_resolvers = {
    key: [(tag, regexp) for tag, regexp in resolvers if tag not in _REPLACED_TAGS]
    for key, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_StrictLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool", re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF")
)
_StrictLoader.add_implicit_resolver(
    "tag:yaml.org,2002:int", re.compile(r"^[-+]?(?:0|[1-9][0-9]*)$"), list("-+0123456789")
)
_StrictLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"^(?:[-+]?(?:[0-9]+\.[0-9]*|\.[0-9]+)(?:[eE][-+]?[0-9]+)?|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$"
    ),
    list("-+0123456789."),
)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config {path}: {exc.strerror}") from None
    except UnicodeDecodeError:
        raise ConfigError(f"config {path} is not valid UTF-8") from None
    try:
        data = yaml.load(text, Loader=_StrictLoader)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from None
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"config {path} must be a mapping at the top level")
    return data


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve(path: Path, depth: int, seen: tuple[Path, ...]) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        raise ConfigError(f"config extends cycle: {' -> '.join(str(p) for p in (*seen, path))}")
    if depth > _MAX_EXTENDS_DEPTH:
        raise ConfigError(f"config extends chain deeper than {_MAX_EXTENDS_DEPTH}")
    data = _read_yaml(path)
    parent = data.pop("extends", None)
    if parent is None:
        return data
    if not isinstance(parent, str) or not parent:
        raise ConfigError(f"`extends` in {path} must be a relative file name")
    return _deep_merge(_resolve(path.parent / parent, depth + 1, (*seen, path)), data)


def load_config_dict(path: Path) -> dict[str, Any]:
    """Return the merged, unvalidated mapping for a config file (following `extends`)."""
    return _resolve(Path(path), 0, ())


def validate_config(data: Mapping[str, Any]) -> AppConfig:
    from pydantic import ValidationError

    try:
        return AppConfig.model_validate(data)
    except ValidationError as exc:
        lines = []
        for error in exc.errors():
            location = ".".join(str(part) for part in error["loc"]) or "<root>"
            lines.append(f"{location}: {error['msg']}")
        raise ConfigError("invalid configuration:\n  " + "\n  ".join(lines)) from None


def load_config(path: Path) -> AppConfig:
    """Load, merge and validate a configuration file."""
    return validate_config(load_config_dict(path))
