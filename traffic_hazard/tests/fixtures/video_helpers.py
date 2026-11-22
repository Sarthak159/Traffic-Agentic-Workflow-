"""Video test helpers: tiny in-memory frames and an independent, non-streaming oracle."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
from traffic_hazard.config import VideoConfig
from traffic_hazard.video.reader import DecodeIssue, InMemoryVideoSource, RawFrame
from traffic_hazard.video.windows import WindowResult, build_windows


def video_cfg(**overrides) -> VideoConfig:
    data = {
        "window_seconds": 5.0,
        "assessment_interval_seconds": 1.0,
        "sample_fps": 4.0,
        "max_frames": 20,
        "max_width": 1280,
        "transport": "frames",
        "startup": "require_full_window",
        "reject_out_of_order": True,
    }
    data.update(overrides)
    return VideoConfig.model_validate(data)


def tiny_image(seed: int, variant: int = 0) -> np.ndarray:
    """An 8x8 image whose pixels encode `seed` so every frame hashes differently."""
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[..., 0] = seed % 256
    image[..., 1] = (seed // 256) % 256
    image[..., 2] = variant
    return image


def raw_frames(times: list[float | None], variant_after: float | None = None) -> list[RawFrame]:
    frames = []
    for i, t in enumerate(times):
        variant = 1 if variant_after is not None and t is not None and t > variant_after else 0
        frames.append(
            RawFrame(time=None if t is None else Fraction(str(t)), image=tiny_image(i, variant))
        )
    return frames


def run(
    items: list[RawFrame | DecodeIssue], cfg: VideoConfig | None = None, source_id: str = "cam"
) -> list[WindowResult]:
    return list(
        build_windows(InMemoryVideoSource(items, origin=Fraction(0)), cfg or video_cfg(), source_id)
    )


def oracle(times_s: list[float], cfg: VideoConfig) -> dict[float, list[float]]:
    """Brute-force selection from the full, sorted frame list, independent of the builder.

    Returns cutoff -> selected frame times for every cutoff the footage reached.
    """
    times = sorted(Fraction(str(t)) for t in times_s)
    window, interval = (
        Fraction(str(cfg.window_seconds)),
        Fraction(str(cfg.assessment_interval_seconds)),
    )
    spacing = 1 / Fraction(str(cfg.sample_fps))
    n_ticks = math.ceil(window / spacing)
    cutoff = math.ceil((times[0] + window) / interval) * interval
    result = {}
    while cutoff <= times[-1]:
        chosen = set()
        for k in range(n_ticks):
            tick = cutoff - k * spacing
            before = [t for t in times if t <= tick]
            if before and before[-1] > cutoff - window:
                chosen.add(before[-1])
        if chosen:
            result[float(cutoff)] = sorted(float(t) for t in chosen)
        cutoff += interval
    return result
