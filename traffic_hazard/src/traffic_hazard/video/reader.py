"""Video sources: decoded frames with presentation timestamps normalized to source seconds.

Every source feeds raw `(timestamp, image)` pairs through `normalize_stream`, the single place that
quantizes time, rejects frames without timestamps, drops duplicate or out-of-order timestamps, and
reports gaps.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

US_PER_S = 1_000_000
DISCONTINUITY_GAP_S = Fraction(1)

IssueKind = Literal[
    "corrupt",
    "missing_timestamp",
    "duplicate_timestamp",
    "out_of_order",
    "discontinuity",
    "before_origin",
]


@dataclass(frozen=True)
class DecodedFrame:
    """One accepted frame. `time_us` is integer microseconds since the source origin."""

    index: int
    time_us: int
    image: np.ndarray  # H x W x 3, uint8, RGB

    @property
    def time(self) -> Fraction:
        return Fraction(self.time_us, US_PER_S)


@dataclass(frozen=True)
class DecodeIssue:
    """A frame-level problem.

    `time_us` is the issue's own source time, or None when unknown (for example an undecodable
    packet). Consumers must not attribute an untimed issue to an earlier frame, because it may
    concern later footage.
    """

    kind: IssueKind
    time_us: int | None
    detail: str


@dataclass(frozen=True)
class RawFrame:
    """Decoder output before normalization; `time` is seconds on the decoder's timeline or None."""

    time: Fraction | None
    image: np.ndarray


class VideoSource(Protocol):
    """Yields accepted frames in strictly increasing time order, interleaved with issues."""

    def __iter__(self) -> Iterator[DecodedFrame | DecodeIssue]: ...


def to_us(seconds: Fraction) -> int:
    """Quantize exact seconds to integer microseconds (round half to even)."""
    return round(seconds * US_PER_S)


def normalize_stream(
    raw: Iterable[RawFrame | DecodeIssue], origin: Fraction | None = None
) -> Iterator[DecodedFrame | DecodeIssue]:
    """Normalize raw decoder output into a strictly increasing frame stream.

    The origin is the container stream start when known, else the first timestamped frame.
    Out-of-order and duplicate timestamps are always dropped (first decoded frame wins), because a
    causal window builder cannot accept a frame earlier than one it has already seen.
    """
    last_us: int | None = None
    index = 0
    for item in raw:
        if isinstance(item, DecodeIssue):
            yield item
            continue
        if item.time is None:
            yield DecodeIssue("missing_timestamp", None, "frame has no presentation timestamp")
            continue
        if origin is None:
            origin = item.time
        time_us = to_us(item.time - origin)
        if time_us < 0:
            yield DecodeIssue(
                "before_origin", None, f"timestamp {float(item.time):.6f}s precedes origin"
            )
            continue
        if last_us is not None and time_us == last_us:
            yield DecodeIssue("duplicate_timestamp", time_us, "duplicate timestamp dropped")
            continue
        if last_us is not None and time_us < last_us:
            yield DecodeIssue(
                "out_of_order", time_us, "timestamp earlier than previous frame dropped"
            )
            continue
        if last_us is not None and Fraction(time_us - last_us, US_PER_S) > DISCONTINUITY_GAP_S:
            yield DecodeIssue(
                "discontinuity", time_us, f"gap of {(time_us - last_us) / US_PER_S:.3f}s"
            )
        image = np.ascontiguousarray(item.image, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            yield DecodeIssue("corrupt", time_us, f"unexpected image shape {image.shape}")
            continue
        yield DecodedFrame(index=index, time_us=time_us, image=image)
        index += 1
        last_us = time_us


@dataclass
class InMemoryVideoSource:
    """Test and synthetic source built from raw frames (timestamps may be missing or disordered)."""

    raw: list[RawFrame | DecodeIssue]
    origin: Fraction | None = None

    def __iter__(self) -> Iterator[DecodedFrame | DecodeIssue]:
        return normalize_stream(self.raw, self.origin)


@dataclass
class PyAVVideoSource:
    """Decode the first video stream of a local file with PyAV, using PTS and stream time base."""

    path: Path

    def __iter__(self) -> Iterator[DecodedFrame | DecodeIssue]:
        import av

        try:
            container = av.open(str(self.path), metadata_errors="ignore")
        except (av.FFmpegError, OSError) as exc:
            raise VideoOpenError(f"cannot open video: {type(exc).__name__}") from None
        try:
            if not container.streams.video:
                raise VideoOpenError("file has no video stream")
            stream = container.streams.video[0]
            time_base = stream.time_base
            if time_base is None:
                raise VideoOpenError("video stream has no time base")
            origin = (
                Fraction(stream.start_time) * time_base if stream.start_time is not None else None
            )
            yield from normalize_stream(self._raw(container, stream, time_base), origin)
        finally:
            container.close()

    @staticmethod
    def _raw(container, stream, time_base: Fraction) -> Iterator[RawFrame | DecodeIssue]:
        import av

        packets = container.demux(stream)
        while True:
            try:
                packet = next(packets)
            except StopIteration:
                return
            except av.FFmpegError as exc:
                yield DecodeIssue("corrupt", None, f"demux stopped: {type(exc).__name__}")
                return
            try:
                frames = packet.decode()
            except av.FFmpegError as exc:
                yield DecodeIssue("corrupt", None, f"packet decode failed: {type(exc).__name__}")
                continue
            for frame in frames:
                if getattr(frame, "corrupt", False):
                    yield DecodeIssue("corrupt", None, "decoder flagged frame as corrupt")
                    continue
                time = Fraction(frame.pts) * time_base if frame.pts is not None else None
                yield RawFrame(time=time, image=frame.to_ndarray(format="rgb24"))


class VideoOpenError(RuntimeError):
    """The file could not be opened as a video; raised before any frame is produced."""
