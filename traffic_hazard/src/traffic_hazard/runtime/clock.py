"""Replay clocks. Source time is footage time, never wall-clock time; durations use a monotonic
clock.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field


class UnpacedClock:
    """Offline replay as fast as possible; never waits."""

    async def wait_until(self, source_time_s: float) -> float:
        """Return how far wall time lags the paced schedule (always 0 when unpaced)."""
        return 0.0


@dataclass
class PacedClock:
    """Replay at `speed` x real time, anchored at the first source time requested.

    `wait_until` sleeps until the wall-clock moment corresponding to `source_time_s` and returns the
    lag in seconds when the caller is already late (it never sleeps a negative amount or catches up
    by skipping).
    """

    speed: float = 1.0
    monotonic: Callable[[], float] = time.monotonic
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    _anchor: tuple[float, float] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not self.speed > 0:
            raise ValueError("speed must be positive")

    async def wait_until(self, source_time_s: float) -> float:
        now = self.monotonic()
        if self._anchor is None:
            self._anchor = (now, source_time_s)
            return 0.0
        wall0, source0 = self._anchor
        if source_time_s < source0:
            raise ValueError("source time moved backwards")
        target = wall0 + (source_time_s - source0) / self.speed
        if target > now:
            await self.sleep(target - now)
            return 0.0
        return now - target


@dataclass
class Stopwatch:
    """Monotonic elapsed-time measurement."""

    monotonic: Callable[[], float] = time.monotonic
    _start: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self._start = self.monotonic()

    def elapsed(self) -> float:
        return self.monotonic() - self._start
