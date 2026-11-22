"""Write a deterministic synthetic development clip (no real footage is used or downloaded).

Usage (from `traffic_hazard/`): uv run python scripts/make_fixture.py --out data/dev/fixture.mp4
"""

from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path

from traffic_hazard.video.synthetic import cfr_times, write_video


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/dev/fixture.mp4"))
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    codec = "libx264" if args.out.suffix == ".mp4" else "ffv1"
    time_base = Fraction(1, 30000) if codec == "libx264" else Fraction(1, 1000)
    path = write_video(
        args.out,
        cfr_times(args.seconds, args.fps),
        codec=codec,
        time_base=time_base,
        size=(640, 360),
    )
    print(f"wrote {path} ({args.seconds} s at {args.fps} FPS, {codec})")


if __name__ == "__main__":
    main()
