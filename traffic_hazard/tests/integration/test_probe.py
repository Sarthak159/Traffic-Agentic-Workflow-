from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

from traffic_hazard.cli import main
from traffic_hazard.config import load_config
from traffic_hazard.probe import MEDIA_PROBES, run_probe, synthetic_probe_window
from traffic_hazard.providers.fake import FakeProvider, FakeResponse, malformed
from traffic_hazard.video.synthetic import cfr_times, write_video

from tests.conftest import CONFIGS

FAKE = str(CONFIGS / "local_fake.yaml")


def _status(report: dict) -> dict[str, str]:
    return {p["name"]: p["status"] for p in report["probes"]}


async def test_fake_probe_passes_harness_checks_but_never_claims_media_capability() -> None:
    cfg = load_config(CONFIGS / "local_fake.yaml")
    provider = FakeProvider(model_id=cfg.model.model_id)
    report = await run_probe(cfg, provider, synthetic_probe_window(cfg), repeats=3)
    status = _status(report)
    assert report["simulated"] is True and "not evidence" in report["notice"]
    assert (
        status["connectivity"] == status["json_assessment"] == status["json_specialist"] == "passed"
    )
    assert all(status[name] == "not_evaluable" for name in MEDIA_PROBES)
    assert status["tool_selection"] == "not_evaluable" and status["native_video"] == "skipped"
    assert report["total_attempts"] == provider.call_count == 5  # 2 JSON probes + 3 latency.
    assert report["latency"]["usage_reported_attempts"] == 0  # Unknown usage stays null.
    assert report["latency"]["request_payload_bytes"] is None


async def test_probe_reports_failures_instead_of_passing() -> None:
    cfg = load_config(CONFIGS / "local_fake.yaml")
    window = synthetic_probe_window(cfg)
    bad_json = FakeProvider(script={"main": malformed()}, model_id=cfg.model.model_id)
    status = _status(await run_probe(cfg, bad_json, window, repeats=1))
    assert status["connectivity"] == "passed" and status["json_assessment"] == "failed"
    assert status["cost_latency"] == "failed"

    down = FakeProvider(script={"main": FakeResponse(kind="error")}, model_id=cfg.model.model_id)
    assert _status(await run_probe(cfg, down, window, repeats=1))["connectivity"] == "failed"

    renamed = FakeProvider(model_id="some-other-model")
    report = await run_probe(cfg, renamed, window, repeats=1)
    assert _status(report)["connectivity"] == "failed"
    assert report["effective_model_ids"] == ["some-other-model"]


def test_probe_cli_with_synthetic_and_video_windows(tmp_path: Path, capsys) -> None:
    out = tmp_path / "probe"
    assert main(["probe", "--config", FAKE, "--out", str(out), "--repeats", "2"]) == 0
    report = json.loads((out / "probe_report.json").read_text())
    assert report["simulated"] and report["total_attempts"] == 4
    assert "NOTICE" in capsys.readouterr().out

    video = write_video(
        tmp_path / "v.mp4", cfr_times(7, 10), codec="libx264", time_base=Fraction(1, 30000)
    )
    assert main(["probe", "--config", FAKE, "--video", str(video), "--source-id", "cam-1"]) == 0
    assert main(["probe", "--config", FAKE, "--video", str(video)]) == 2  # --source-id required.
    short = write_video(
        tmp_path / "s.mp4", cfr_times(2, 10), codec="libx264", time_base=Fraction(1, 30000)
    )
    assert main(["probe", "--config", FAKE, "--video", str(short), "--source-id", "cam-1"]) == 2
    assert main(["probe", "--config", FAKE, "--repeats", "0"]) == 2
    assert main(["probe", "--config", FAKE, "--out", str(out)]) == 2  # Non-empty output refused.


def test_probe_without_video_works_for_any_valid_interval(tmp_path: Path) -> None:
    cfg_path = tmp_path / "interval.yaml"
    cfg_path.write_text(
        f"extends: {CONFIGS / 'defaults.yaml'}\nvideo:\n  assessment_interval_seconds: 3.0\n"
    )
    assert main(["probe", "--config", str(cfg_path), "--repeats", "1"]) == 0


def test_refused_probe_leaves_no_output_directory(tmp_path: Path) -> None:
    short = write_video(
        tmp_path / "s.mp4", cfr_times(2, 10), codec="libx264", time_base=Fraction(1, 30000)
    )
    out = tmp_path / "never"
    assert (
        main(
            [
                "probe",
                "--config",
                FAKE,
                "--video",
                str(short),
                "--source-id",
                "c",
                "--out",
                str(out),
            ]
        )
        == 2
    )
    assert not out.exists()
