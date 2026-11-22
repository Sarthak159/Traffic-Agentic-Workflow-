"""Deterministic synthetic intersection-like clips for tests and development (no real footage)."""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def synthetic_image(
    time_s: float, width: int = 160, height: int = 96, variant: int = 0
) -> np.ndarray:
    """A grey road with a crosswalk, a 'vehicle' block moving right and a 'pedestrian' dot moving
    down.

    The frame time is burnt in as text so contact sheets can be checked by eye. `variant` changes
    colors so tests can create footage that differs only after a chosen time.
    """
    # Draw on a small canvas and scale up, so shapes and the burnt-in time stay legible at any size.
    base_w, base_h = 160, 96
    out_w, out_h = width, height
    width, height = base_w, base_h
    img = Image.new("RGB", (width, height), (90, 90, 90))
    draw = ImageDraw.Draw(img)
    cx = width // 2
    for y in range(8, height - 8, 8):
        draw.rectangle([cx - 10, y, cx + 10, y + 3], fill=(230, 230, 230))
    vx = int((time_s * 12) % (width + 30)) - 30
    color = (200, 40, 40) if variant == 0 else (40, 40, 200)
    draw.rectangle([vx, height // 2 - 8, vx + 26, height // 2 + 8], fill=color)
    py = int((time_s * 6) % height)
    draw.ellipse([cx - 3, py - 3, cx + 3, py + 3], fill=(250, 220, 0))
    draw.text((2, 2), f"t={time_s:.3f}", fill=(255, 255, 255))
    if (out_w, out_h) != (base_w, base_h):
        img = img.resize((out_w, out_h), Image.NEAREST)
    return np.asarray(img, dtype=np.uint8)


def write_video(
    path: Path,
    times_s: Sequence[float],
    *,
    codec: str = "ffv1",
    time_base: Fraction = Fraction(1, 1000),
    size: tuple[int, int] = (160, 96),
    variant_after_s: float | None = None,
) -> Path:
    """Encode frames at exact presentation times (variable frame rate is allowed).

    `ffv1` is intra-only and deterministic, so decoded pixels (and hashes) of frames before a
    modification point are unaffected by later changes; inter-frame codecs do not give that
    guarantee.
    """
    import av

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = size
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream(codec, rate=30)
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv444p" if codec == "ffv1" else "yuv420p"
        stream.time_base = time_base
        stream.codec_context.time_base = time_base
        for t in times_s:
            variant = 1 if variant_after_s is not None and t > variant_after_s else 0
            frame = av.VideoFrame.from_ndarray(
                synthetic_image(t, width, height, variant), format="rgb24"
            )
            frame.pts = round(Fraction(str(t)) / time_base)
            frame.time_base = time_base
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def cfr_times(duration_s: float, fps: float, start_s: float = 0.0) -> list[float]:
    count = int(round(duration_s * fps))
    return [round(start_s + i / fps, 6) for i in range(count)]
