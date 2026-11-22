from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

from traffic_hazard.cli import main
from traffic_hazard.contracts import ObservationWindow
from traffic_hazard.video.synthetic import cfr_times, write_video

from tests.conftest import CONFIGS

FAKE = str(CONFIGS / "local_fake.yaml")


def _clip(tmp_path: Path, seconds: float, name: str = "clip.mp4") -> Path:
    times = cfr_times(seconds, 30)
    return write_video(tmp_path / name, times, codec="libx264", time_base=Fraction(1, 30000))


def _windows(video: Path, out: Path, *extra: str, source_id: str = "camera-dev-01") -> int:
    args = ["windows", "--video", str(video), "--config", FAKE, "--out", str(out)]
    return main([*args, "--source-id", source_id, *extra])


def _read_windows(out: Path) -> list[ObservationWindow]:
    lines = (out / "windows.jsonl").read_text().splitlines()
    return [ObservationWindow.model_validate_json(line) for line in lines]


def _summary(out: Path) -> dict:
    return json.loads((out / "extraction_summary.json").read_text())


def test_windows_command_writes_manifest_media_and_sheets(tmp_path: Path) -> None:
    out = tmp_path / "run"
    assert _windows(_clip(tmp_path, 9), out, "--save-frames", "--contact-sheet") == 0
    windows = _read_windows(out)
    assert [w.cutoff_s for w in windows] == [5.0, 6.0, 7.0, 8.0]
    summary = _summary(out)
    assert summary["status"] == "ok" and summary["windows_emitted"] == 4
    assert {w.source_id for w in windows} == {"camera-dev-01"}
    assert summary["source"]["file_name"] == "clip.mp4" and len(summary["source"]["sha256"]) == 64
    refs = {f.local_ref for w in windows for f in w.frames}
    assert all((out / ref).is_file() for ref in refs)
    sheets = summary["contact_sheets"]
    assert len(sheets) == 2 and all((out / s).is_file() for s in sheets)


def test_windows_command_is_deterministic_and_overwrite_replaces_own_artifacts(
    tmp_path: Path,
) -> None:
    video = _clip(tmp_path, 7)
    a, b = tmp_path / "a", tmp_path / "b"
    assert _windows(video, a, "--contact-sheet") == 0
    assert _windows(video, b) == 0
    assert (a / "windows.jsonl").read_bytes() == (b / "windows.jsonl").read_bytes()
    assert _windows(video, a) == 2  # A non-empty output directory is refused.

    (a / "notes.txt").write_text("user file")
    assert _windows(video, a, "--overwrite") == 0
    # Overwrite replaces this command's artifacts instead of appending, and keeps unrelated files.
    assert (a / "windows.jsonl").read_bytes() == (b / "windows.jsonl").read_bytes()
    assert not (a / "contact_sheets").exists()
    assert (a / "notes.txt").read_text() == "user file"


def test_short_clip_reports_insufficient_history(tmp_path: Path) -> None:
    out = tmp_path / "short"
    assert _windows(_clip(tmp_path, 3), out) == 1
    summary = _summary(out)
    assert summary["status"] == "insufficient_history" and summary["windows_emitted"] == 0
    assert (out / "windows.jsonl").read_text() == ""


def test_bad_inputs_are_reported(tmp_path: Path) -> None:
    assert _windows(tmp_path / "missing.mp4", tmp_path / "x") == 2
    assert _windows(_clip(tmp_path, 6), tmp_path / "y", source_id="../escape") == 2
    not_a_dir = tmp_path / "file.txt"
    not_a_dir.write_text("x")
    assert _windows(_clip(tmp_path, 6, "c2.mp4"), not_a_dir) == 2
    bogus = tmp_path / "bogus.mp4"
    bogus.write_text("not a video")
    assert _windows(bogus, tmp_path / "z") == 1
    assert _summary(tmp_path / "z")["status"] == "failed"


def test_cli_window_ids_ignore_footage_after_the_cutoff(tmp_path: Path) -> None:
    times = cfr_times(10, 10)
    a = write_video(tmp_path / "a.mkv", times)
    b = write_video(tmp_path / "b.mkv", times, variant_after_s=8.0)
    assert _windows(a, tmp_path / "ra") == 0 and _windows(b, tmp_path / "rb") == 0
    wa, wb = _read_windows(tmp_path / "ra"), _read_windows(tmp_path / "rb")
    assert [w for w in wa if w.cutoff_s <= 8.0] == [w for w in wb if w.cutoff_s <= 8.0]
    assert (
        _summary(tmp_path / "ra")["source"]["sha256"]
        != _summary(tmp_path / "rb")["source"]["sha256"]
    )


def test_source_id_is_required(tmp_path: Path, capsys) -> None:
    import pytest

    with pytest.raises(SystemExit):
        main(["windows", "--video", "v.mp4", "--config", FAKE, "--out", str(tmp_path / "o")])
