from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest
from traffic_hazard.video.reader import DecodedFrame, DecodeIssue, PyAVVideoSource, VideoOpenError
from traffic_hazard.video.synthetic import cfr_times, write_video
from traffic_hazard.video.windows import WindowBuilder


def _split(path: Path) -> tuple[list[DecodedFrame], list[DecodeIssue]]:
    items = list(PyAVVideoSource(path))
    return [i for i in items if isinstance(i, DecodedFrame)], [
        i for i in items if isinstance(i, DecodeIssue)
    ]


@pytest.mark.parametrize(
    ("codec", "ext", "time_base", "expected_us"),
    [
        # Matroska stores millisecond timestamps, so 1/30 s is honestly 33 ms there.
        ("ffv1", "mkv", Fraction(1, 1000), [0, 33000, 67000]),
        ("libx264", "mp4", Fraction(1, 30000), [0, 33333, 66667]),
        ("mpeg4", "mp4", Fraction(1, 30000), [0, 33333, 66667]),
    ],
)
def test_cfr_timestamps_come_from_pts(
    tmp_path: Path, codec: str, ext: str, time_base: Fraction, expected_us: list[int]
) -> None:
    times = [i / 30 for i in range(90)]
    path = write_video(tmp_path / f"c.{ext}", times, codec=codec, time_base=time_base)
    frames, issues = _split(path)
    assert len(frames) == 90 and issues == []
    assert [f.time_us for f in frames[:3]] == expected_us
    assert frames[0].image.shape == (96, 160, 3)


def test_variable_frame_rate_uses_real_timestamps(tmp_path: Path) -> None:
    times = [0.0, 0.04, 0.5, 0.52, 1.3, 2.9, 3.0]
    frames, issues = _split(write_video(tmp_path / "vfr.mkv", times))
    assert [f.time_us / 1e6 for f in frames] == times
    assert [i.kind for i in issues] == ["discontinuity"]


def test_stream_start_offset_is_normalized(tmp_path: Path) -> None:
    frames, _ = _split(write_video(tmp_path / "late.mkv", cfr_times(2, 10, start_s=3.0)))
    assert frames[0].time_us == 0 and frames[-1].time_us == 1_900_000


def test_truncated_file_ends_cleanly(tmp_path: Path, fake_cfg) -> None:
    path = write_video(tmp_path / "full.mkv", cfr_times(10, 10))
    data = path.read_bytes()
    truncated = tmp_path / "truncated.mkv"
    truncated.write_bytes(data[: int(len(data) * 0.7)])
    frames, _ = _split(truncated)
    assert 0 < len(frames) < 100
    builder = WindowBuilder(fake_cfg.video, "cam")
    windows = [
        w for item in PyAVVideoSource(truncated) for w in builder.feed(item)
    ] + builder.finish()
    assert windows, "a 7 s readable prefix should still yield windows"
    assert all(w.window.cutoff_s <= frames[-1].time_us / 1e6 for w in windows)


def test_corrupted_bytes_do_not_crash_or_break_causality(tmp_path: Path, fake_cfg) -> None:
    path = write_video(tmp_path / "ok.mp4", cfr_times(10, 25), codec="libx264")
    data = bytearray(path.read_bytes())
    for offset in range(len(data) // 3, len(data) // 3 + 4000, 97):
        data[offset] = (data[offset] + 91) % 256
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(bytes(data))
    builder = WindowBuilder(fake_cfg.video, "cam")
    for item in PyAVVideoSource(corrupt):
        for result in builder.feed(item):
            for frame in result.window.frames:
                assert result.window.start_s < frame.source_time_s <= result.window.cutoff_s
    builder.finish()
    assert builder.frames_seen > 0


def test_non_video_file_is_rejected_before_any_frame(tmp_path: Path) -> None:
    bogus = tmp_path / "notes.mp4"
    bogus.write_text("not a video")
    with pytest.raises(VideoOpenError):
        list(PyAVVideoSource(bogus))
    with pytest.raises(VideoOpenError):
        list(PyAVVideoSource(tmp_path / "missing.mp4"))
