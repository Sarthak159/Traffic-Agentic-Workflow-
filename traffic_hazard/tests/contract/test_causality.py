"""Causality contract: a window for cutoff t depends only on footage at or before t."""

from __future__ import annotations

from pathlib import Path

import pytest
from traffic_hazard.video.reader import PyAVVideoSource
from traffic_hazard.video.synthetic import cfr_times, write_video
from traffic_hazard.video.windows import build_windows

from tests.fixtures.video_helpers import raw_frames, run


def _windows(path: Path, cfg, source_id: str = "cam"):
    return [r.window for r in build_windows(PyAVVideoSource(path), cfg, source_id)]


def test_frames_after_cutoff_do_not_change_window_hash_in_memory() -> None:
    times = [round(i / 10, 6) for i in range(150)]
    original = run(raw_frames(times))
    modified = run(raw_frames(times, variant_after=8.0))
    same = [a.window for a in original if a.window.cutoff_s <= 8.0]
    assert same == [b.window for b in modified if b.window.cutoff_s <= 8.0]
    later_a = {a.window.cutoff_s: a.window.window_id for a in original if a.window.cutoff_s > 8.0}
    later_b = {b.window.cutoff_s: b.window.window_id for b in modified if b.window.cutoff_s > 8.0}
    assert later_a.keys() == later_b.keys() and all(later_a[k] != later_b[k] for k in later_a)


@pytest.mark.parametrize("cut", [5.0, 6.5, 9.0, 12.0])
def test_truncating_the_stream_after_t_reproduces_the_window_for_t(cut: float) -> None:
    times = [round(i / 7, 6) for i in range(100)]
    full = {r.window.cutoff_s: r.window for r in run(raw_frames(times))}
    prefix = {r.window.cutoff_s: r.window for r in run(raw_frames([t for t in times if t <= cut]))}
    assert prefix, "prefix should contain at least one window"
    for cutoff, window in prefix.items():
        assert full[cutoff] == window


def test_all_selected_frames_lie_in_the_causal_interval(cfg_video=None) -> None:
    times = [round(i * 0.137, 6) for i in range(200)]
    for result in run(raw_frames(times)):
        window = result.window
        for frame in window.frames:
            assert window.start_s < frame.source_time_s <= window.cutoff_s


def test_future_modification_of_a_real_file_leaves_earlier_windows_unchanged(
    tmp_path: Path, fake_cfg
) -> None:
    times = cfr_times(12, 10)
    a = write_video(tmp_path / "a.mkv", times)
    b = write_video(tmp_path / "b.mkv", times, variant_after_s=8.0)
    wa, wb = _windows(a, fake_cfg.video), _windows(b, fake_cfg.video)
    assert [w for w in wa if w.cutoff_s <= 8.0] == [w for w in wb if w.cutoff_s <= 8.0]
    assert [w.window_id for w in wa if w.cutoff_s > 8.0] != [
        w.window_id for w in wb if w.cutoff_s > 8.0
    ]


def test_repeat_extraction_is_identical(tmp_path: Path, fake_cfg) -> None:
    path = write_video(tmp_path / "clip.mp4", cfr_times(9, 30), codec="libx264")
    assert _windows(path, fake_cfg.video) == _windows(path, fake_cfg.video)


def test_window_id_depends_on_source_id_and_sampling(tmp_path: Path, fake_cfg) -> None:
    path = write_video(tmp_path / "clip.mkv", cfr_times(7, 10))
    base = _windows(path, fake_cfg.video)[0]
    other_source = _windows(path, fake_cfg.video, source_id="cam2")[0]
    smaller = _windows(path, fake_cfg.video.model_copy(update={"max_width": 64}))[0]
    assert len({base.window_id, other_source.window_id, smaller.window_id}) == 3
