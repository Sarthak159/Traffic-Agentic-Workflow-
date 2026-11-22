from __future__ import annotations

import random
from fractions import Fraction

import pytest
from traffic_hazard.video.reader import DecodedFrame, DecodeIssue, RawFrame
from traffic_hazard.video.windows import WindowBuilder

from tests.fixtures.video_helpers import oracle, raw_frames, run, tiny_image, video_cfg


def _times(result) -> list[float]:
    return [f.source_time_s for f in result.window.frames]


def test_cfr_4fps_windows_are_full_and_on_the_cutoff_grid() -> None:
    times = [i / 4 for i in range(40)]  # 0.00 .. 9.75 s
    results = run(raw_frames(times))
    assert [r.window.cutoff_s for r in results] == [5.0, 6.0, 7.0, 8.0, 9.0]  # 10.0 never reached.
    for r in results:
        assert len(r.window.frames) == 20
        assert _times(r) == [r.window.cutoff_s - 4.75 + i / 4 for i in range(20)]
        assert r.window.quality_flags == ()
        assert r.window.start_s == r.window.cutoff_s - 5.0


def test_30fps_selects_latest_frame_at_or_before_each_tick() -> None:
    times = [round(i / 30, 6) for i in range(30 * 8)]
    results = run(raw_frames(times))
    assert {r.window.cutoff_s: _times(r) for r in results} == oracle(times, video_cfg())
    assert all(len(r.window.frames) == 20 for r in results)


@pytest.mark.parametrize("seed", range(8))
def test_variable_frame_rate_matches_brute_force_oracle(seed: int) -> None:
    rng = random.Random(seed)
    t, times = 0.0, []
    while t < 20:
        times.append(round(t, 6))
        t += rng.choice([0.01, 0.033, 0.1, 0.25, 0.4, 0.9, 1.7, 3.0])
    cfg = video_cfg(
        sample_fps=rng.choice([2.0, 3.0, 4.0]),
        assessment_interval_seconds=rng.choice([0.5, 1.0, 1.5]),
    )
    results = run(raw_frames(times), cfg)
    expected = oracle(times, cfg)
    assert {r.window.cutoff_s: _times(r) for r in results} == expected
    for r in results:
        assert len({f.frame_id for f in r.window.frames}) == len(r.window.frames)


def test_non_integer_tick_count() -> None:
    cfg = video_cfg(sample_fps=3.0)
    assert cfg.ticks_per_window == 15
    results = run(raw_frames([round(i / 30, 6) for i in range(300)]), cfg)
    assert all(len(r.window.frames) == 15 for r in results)


def test_short_clip_reports_insufficient_history_without_padding() -> None:
    builder = WindowBuilder(video_cfg(), "cam")
    from traffic_hazard.video.reader import normalize_stream

    out = []
    for item in normalize_stream(
        raw_frames([i / 4 for i in range(16)]), Fraction(0)
    ):  # 0 .. 3.75 s
        out += builder.feed(item)
    out += builder.finish()
    assert out == [] and builder.skipped == []
    assert builder.last_frame_us == 3_750_000


def test_first_window_needs_full_history_from_first_frame() -> None:
    times = [2.3 + i / 4 for i in range(40)]
    results = run(raw_frames(times))
    assert results[0].window.cutoff_s == 8.0  # ceil(2.3 + 5) on the 1 s grid.


def test_gaps_are_flagged_and_empty_windows_skipped() -> None:
    times = [i / 4 for i in range(25)] + [12 + i / 4 for i in range(16)]  # 0..6 s, then 12..15.75 s
    builder = WindowBuilder(video_cfg(), "cam")
    from traffic_hazard.video.reader import normalize_stream

    results = []
    for item in normalize_stream(raw_frames(times), Fraction(0)):
        results += builder.feed(item)
    results += builder.finish()
    by_cutoff = {r.window.cutoff_s: r.window for r in results}
    assert by_cutoff[6.0].quality_flags == ()
    # Ticks 6.25..7.0 can only reuse the frame at 6.0 s: repeated and stale, but inside the window.
    assert by_cutoff[7.0].quality_flags == ("repeated_frame_ticks:4", "stale_ticks:4")
    assert (
        "gap_ticks:19" in by_cutoff[12.0].quality_flags
    )  # Only the 12.0 s frame is inside (7, 12].
    assert {s.cutoff_s for s in builder.skipped} == {11.0}  # (6, 11] contains no frame.
    assert "decode_issue:discontinuity" in by_cutoff[12.0].quality_flags
    assert len(by_cutoff[12.0].frames) == 1  # Only the frame at 12.0 is inside (7, 12].
    assert builder.issues["discontinuity"] == 1


def test_sparse_frames_are_deduplicated_and_flagged() -> None:
    times = [float(i) for i in range(12)]  # 1 FPS source sampled at 4 FPS.
    results = run(raw_frames(times))
    window = results[0].window
    assert [f.source_time_s for f in window.frames] == [1.0, 2.0, 3.0, 4.0, 5.0]
    # Ticks 0.25..0.75 can only see the frame at 0.0 s, which is outside (0, 5].
    assert window.quality_flags == ("gap_ticks:3", "repeated_frame_ticks:12", "stale_ticks:12")


def test_duplicate_out_of_order_and_missing_timestamps_are_dropped() -> None:
    times = [i / 4 for i in range(30)]
    items = raw_frames(times)
    items.insert(10, RawFrame(time=Fraction(9, 4), image=tiny_image(999)))  # Duplicate of 2.25 s.
    items.insert(20, RawFrame(time=Fraction(1), image=tiny_image(998)))  # Out of order.
    items.insert(5, RawFrame(time=None, image=tiny_image(997)))  # Missing timestamp.
    builder = WindowBuilder(video_cfg(), "cam")
    from traffic_hazard.video.reader import normalize_stream

    results = []
    for item in normalize_stream(items, Fraction(0)):
        results += builder.feed(item)
    results += builder.finish()
    assert {r.window.cutoff_s: _times(r) for r in results} == oracle(times, video_cfg())
    assert dict(builder.issues) == {
        "duplicate_timestamp": 1,
        "out_of_order": 1,
        "missing_timestamp": 1,
    }
    clean = {r.window.cutoff_s: r.window.frames for r in run(raw_frames(times))}
    # Dropped frames never replace the first decoded frame at a timestamp.
    assert {r.window.cutoff_s: r.window.frames for r in results} == clean


def test_untimed_issue_after_cutoff_does_not_change_that_window() -> None:
    times = [i / 4 for i in range(30)]
    base = run(raw_frames(times))
    items = raw_frames(times)
    items.insert(21, DecodeIssue("corrupt", None, "bad packet"))  # After the frame at 5.0 s.
    with_issue = run(items)
    assert with_issue[0].window == base[0].window  # Cutoff 5.0 is unaffected.
    assert "decode_issue:corrupt" in with_issue[1].window.quality_flags  # It belongs to (1, 6].


def test_tick_exactly_on_frame_time_selects_that_frame() -> None:
    times = [0.0, 0.1, 4.999999, 5.0, 5.000001, 7.0]
    results = run(raw_frames(times))
    assert _times(results[0])[-1] == 5.0


def test_builder_rejects_misuse() -> None:
    builder = WindowBuilder(video_cfg(), "cam")
    builder.feed(DecodedFrame(0, 1_000_000, tiny_image(0)))
    with pytest.raises(ValueError, match="increasing"):
        builder.feed(DecodedFrame(1, 1_000_000, tiny_image(1)))
    builder.finish()
    with pytest.raises(RuntimeError):
        builder.feed(DecodedFrame(2, 2_000_000, tiny_image(2)))


def test_retained_state_is_bounded_on_long_streams() -> None:
    builder = WindowBuilder(video_cfg(), "cam")
    peak = 0
    for i in range(30 * 120):
        builder.feed(DecodedFrame(i, round(i * 1_000_000 / 30), tiny_image(i)))
        retained = {f.index for f in builder._selection.values() if f is not None}
        peak = max(peak, len(builder._selection), len(retained), len(builder._encoded))
    assert peak <= 2 * builder.cfg.ticks_per_window


def test_resize_preserves_aspect_and_changes_hash_version() -> None:
    import numpy as np
    from traffic_hazard.video.encoding import encode_frame, resize_to_width

    image = np.zeros((720, 1920, 3), dtype=np.uint8)
    assert resize_to_width(image, 1280).shape == (480, 1280, 3)
    assert resize_to_width(image[:10, :10], 1280).shape == (10, 10, 3)  # Never upscales.
    a, b = encode_frame(image, 1280), encode_frame(image, 640)
    assert a.content_sha256 != b.content_sha256
