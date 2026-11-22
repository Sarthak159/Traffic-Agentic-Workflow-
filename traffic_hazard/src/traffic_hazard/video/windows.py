"""Streaming, causal construction of immutable observation windows.

For a cutoff `t`, sampling ticks are `t - k * spacing` for `k = 0 .. N-1`, all inside `(t - W, t]`.
Each tick selects the latest accepted frame at or before it; a selection at or before `t - W` is a
gap. Repeated selections of one frame are deduplicated. Cutoffs lie on the grid `n * interval` of
source time and the first one requires a full window of history after the first frame. A window is
emitted only once a frame later than `t` arrives (or the stream ends at or after `t`), and its
content depends only on frames at or before `t`.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from ..config import VideoConfig
from ..contracts import FrameRef, ObservationWindow
from ..hashing import fingerprint
from .encoding import ENCODING_VERSION, EncodedFrame, encode_frame
from .reader import US_PER_S, DecodedFrame, DecodeIssue


def frame_id_for(time_us: int) -> str:
    return f"f-{time_us // US_PER_S}.{time_us % US_PER_S:06d}"


def seconds(value: Fraction) -> float:
    return round(float(value), 6)


@dataclass(frozen=True)
class SkippedCutoff:
    cutoff_s: float
    reason: str


@dataclass(frozen=True)
class WindowResult:
    """An emitted window plus the resized pixels of its frames, keyed by content hash."""

    window: ObservationWindow
    images: dict[str, np.ndarray]


@dataclass
class _Pending:
    cutoff: Fraction
    ticks: tuple[Fraction, ...]


@dataclass
class WindowBuilder:
    """Feed frames in time order with `feed`, then call `finish` at end of stream.

    Retained state is bounded by the number of distinct pending sampling ticks, not the stream
    length.
    """

    cfg: VideoConfig
    source_id: str
    issues: Counter[str] = field(default_factory=Counter, init=False)
    skipped: list[SkippedCutoff] = field(default_factory=list, init=False)
    frames_seen: int = field(default=0, init=False)
    first_frame_us: int | None = field(default=None, init=False)
    last_frame_us: int | None = field(default=None, init=False)
    _prev: DecodedFrame | None = field(default=None, init=False)
    _next_cutoff: Fraction | None = field(default=None, init=False)
    _pending: list[_Pending] = field(default_factory=list, init=False)
    _selection: dict[Fraction, DecodedFrame | None] = field(default_factory=dict, init=False)
    _encoded: dict[int, EncodedFrame] = field(default_factory=dict, init=False)
    _issue_times: list[tuple[int, str]] = field(default_factory=list, init=False)
    _untimed: list[str] = field(default_factory=list, init=False)
    _finished: bool = field(default=False, init=False)

    def _ticks(self, cutoff: Fraction) -> tuple[Fraction, ...]:
        spacing = self.cfg.tick_spacing
        return tuple(cutoff - k * spacing for k in range(self.cfg.ticks_per_window - 1, -1, -1))

    def _sampling_fingerprint(self) -> dict[str, object]:
        return {
            "window_seconds": str(self.cfg.window),
            "sample_fps": str(Fraction(str(self.cfg.sample_fps))),
            "max_frames": self.cfg.max_frames,
            "max_width": self.cfg.max_width,
            "encoding_version": ENCODING_VERSION,
            "selection": "latest-at-or-before-tick-v1",
        }

    def feed(self, item: DecodedFrame | DecodeIssue) -> list[WindowResult]:
        """Consume one stream item and return the windows that became complete."""
        if self._finished:
            raise RuntimeError("WindowBuilder.feed called after finish")
        if isinstance(item, DecodeIssue):
            self.issues[item.kind] += 1
            if item.time_us is not None:
                self._issue_times.append((item.time_us, item.kind))
            else:
                # Untimed issues happened after the previous frame; attribute them to the next one.
                self._untimed.append(item.kind)
            return []
        if self.last_frame_us is not None and item.time_us <= self.last_frame_us:
            raise ValueError("frames must be fed in strictly increasing time order")
        self.frames_seen += 1
        self._issue_times.extend((item.time_us, kind) for kind in self._untimed)
        self._untimed.clear()
        if self.first_frame_us is None:
            self.first_frame_us = item.time_us
            first_eligible = Fraction(item.time_us, US_PER_S) + self.cfg.window
            self._next_cutoff = math.ceil(first_eligible / self.cfg.interval) * self.cfg.interval
        self.last_frame_us = item.time_us
        now = item.time
        assert self._next_cutoff is not None
        # Register every cutoff whose window has started; its ticks are all later than the previous
        # frame.
        while self._next_cutoff - self.cfg.window < now:
            self._register(self._next_cutoff)
            self._next_cutoff += self.cfg.interval
        # Ticks earlier than this frame are now resolved: the previous frame is the latest at or
        # before them.
        for tick, chosen in self._selection.items():
            if chosen is None and tick < now and self._prev is not None:
                self._selection[tick] = self._prev
        self._prev = item
        return self._emit_ready(lambda cutoff: cutoff < now)

    def finish(self) -> list[WindowResult]:
        """End of stream: resolve ticks up to the last frame; emit cutoffs the footage reached."""
        if self._finished:
            return []
        self._finished = True
        if self._prev is None:
            return []
        last = self._prev.time
        for tick, chosen in self._selection.items():
            if chosen is None and tick >= last:
                self._selection[tick] = self._prev
        results = self._emit_ready(lambda cutoff: cutoff <= last)
        self._pending.clear()
        return results

    def _register(self, cutoff: Fraction) -> None:
        ticks = self._ticks(cutoff)
        self._pending.append(_Pending(cutoff, ticks))
        for tick in ticks:
            self._selection.setdefault(tick, None)

    def _emit_ready(self, ready) -> list[WindowResult]:
        results: list[WindowResult] = []
        while self._pending and ready(self._pending[0].cutoff):
            pending = self._pending.pop(0)
            result = self._build(pending)
            if result is not None:
                results.append(result)
            self._prune()
        return results

    def _prune(self) -> None:
        earliest = min((p.ticks[0] for p in self._pending), default=None)
        for tick in [t for t in self._selection if earliest is None or t < earliest]:
            del self._selection[tick]
        keep_indices = {f.index for f in self._selection.values() if f is not None}
        self._encoded = {i: e for i, e in self._encoded.items() if i in keep_indices}
        horizon_us = round((earliest if earliest is not None else Fraction(0)) * US_PER_S)
        horizon_us -= round(self.cfg.window * US_PER_S)
        self._issue_times = [(t, k) for t, k in self._issue_times if t > horizon_us]

    def _build(self, pending: _Pending) -> WindowResult | None:
        start = pending.cutoff - self.cfg.window
        chosen: dict[int, DecodedFrame] = {}
        gaps = repeats = stale = 0
        for tick in pending.ticks:
            frame = self._selection.get(tick)
            if frame is None or frame.time <= start:
                gaps += 1
                continue
            # A frame at least one sampling interval older than its tick is held over a source gap.
            if tick - frame.time >= self.cfg.tick_spacing:
                stale += 1
            if frame.index in chosen:
                repeats += 1
            else:
                chosen[frame.index] = frame
        if not chosen:
            self.skipped.append(SkippedCutoff(seconds(pending.cutoff), "no_frames_in_window"))
            return None
        frames = sorted(chosen.values(), key=lambda f: f.time_us)
        refs: list[FrameRef] = []
        images: dict[str, np.ndarray] = {}
        for frame in frames:
            encoded = self._encoded.get(frame.index)
            if encoded is None:
                encoded = self._encoded[frame.index] = encode_frame(frame.image, self.cfg.max_width)
            refs.append(
                FrameRef(
                    frame_id=frame_id_for(frame.time_us),
                    source_time_s=frame.time_us / US_PER_S,
                    content_sha256=encoded.content_sha256,
                    local_ref=f"media/{encoded.content_sha256}.jpg",
                )
            )
            images[encoded.content_sha256] = encoded.image
        flags = []
        if gaps:
            flags.append(f"gap_ticks:{gaps}")
        if repeats:
            flags.append(f"repeated_frame_ticks:{repeats}")
        if stale:
            flags.append(f"stale_ticks:{stale}")
        start_us, cutoff_us = round(start * US_PER_S), round(pending.cutoff * US_PER_S)
        issue_kinds = sorted({k for t, k in self._issue_times if start_us < t <= cutoff_us})
        flags.extend(f"decode_issue:{kind}" for kind in issue_kinds)
        identity = {
            "source_id": self.source_id,
            "cutoff_us": cutoff_us,
            "sampling": self._sampling_fingerprint(),
            "frames": [(r.frame_id, r.content_sha256) for r in refs],
            "quality_flags": flags,
        }
        window = ObservationWindow(
            source_id=self.source_id,
            window_id="w-" + fingerprint(identity)[:24],
            start_s=seconds(start),
            cutoff_s=seconds(pending.cutoff),
            frames=tuple(refs),
            quality_flags=tuple(flags),
        )
        return WindowResult(window=window, images=images)


def build_windows(
    items: Iterable[DecodedFrame | DecodeIssue], cfg: VideoConfig, source_id: str
) -> Iterator[WindowResult]:
    """Convenience generator over a whole stream."""
    builder = WindowBuilder(cfg, source_id)
    for item in items:
        yield from builder.feed(item)
    yield from builder.finish()
