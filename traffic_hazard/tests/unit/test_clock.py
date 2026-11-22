from __future__ import annotations

import pytest
from traffic_hazard.runtime.clock import PacedClock, Stopwatch, UnpacedClock


class FakeTime:
    def __init__(self) -> None:
        self.now = 100.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


async def test_paced_clock_waits_and_reports_lag() -> None:
    t = FakeTime()
    clock = PacedClock(speed=2.0, monotonic=t.monotonic, sleep=t.sleep)
    assert await clock.wait_until(5.0) == 0.0  # Anchors source 5.0 at wall 100.
    assert await clock.wait_until(7.0) == 0.0 and t.slept == [1.0]  # 2 s of source at 2x.
    t.now += 3.0  # Processing fell behind.
    assert await clock.wait_until(8.0) == pytest.approx(2.5)
    assert t.slept == [1.0]  # Never sleeps when late.
    with pytest.raises(ValueError):
        await clock.wait_until(1.0)


async def test_unpaced_clock_and_stopwatch() -> None:
    assert await UnpacedClock().wait_until(1e6) == 0.0
    t = FakeTime()
    watch = Stopwatch(monotonic=t.monotonic)
    t.now += 1.5
    assert watch.elapsed() == 1.5
    with pytest.raises(ValueError):
        PacedClock(speed=0)
