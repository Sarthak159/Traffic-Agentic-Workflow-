from __future__ import annotations

from pathlib import Path

import pytest
from traffic_hazard.config import ConfigError, load_config, validate_config

from tests.conftest import CONFIGS


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def _overlay(tmp_path: Path, body: str) -> Path:
    (tmp_path / "defaults.yaml").write_text((CONFIGS / "defaults.yaml").read_text())
    return _write(tmp_path, "cfg.yaml", "extends: defaults.yaml\n" + body)


def test_shipped_configs_validate() -> None:
    defaults = load_config(CONFIGS / "defaults.yaml")
    fake = load_config(CONFIGS / "local_fake.yaml")
    assert fake.model.provider == "fake"
    assert fake.logging.cache_mode == "off"  # YAML 1.1 would have parsed `off` as False.
    assert fake.logging.sdk_remote_tracing is False
    assert defaults.fingerprint() == fake.fingerprint()  # local_fake restates defaults only.


def test_fingerprint_changes_with_settings(tmp_path: Path) -> None:
    changed = load_config(_overlay(tmp_path, "video:\n  sample_fps: 2.0\n"))
    assert changed.fingerprint() != load_config(CONFIGS / "defaults.yaml").fingerprint()
    assert (
        changed.video.sample_fps == 2.0 and changed.video.max_frames == 20
    )  # Deep merge keeps siblings.


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("video:\n  fps: 4\n", "video.fps"),
        ("surprise: 1\n", "surprise"),
        ("workflow:\n  fixed_specialists: [temporal]\n", "not in available_specialists"),
        ("video:\n  sample_fps: 10.0\n", "exceeding max_frames"),
        (
            "workflow:\n  max_total_model_calls: 2\n  max_specialist_calls: 1\n",
            "max_total_model_calls >= 3",
        ),
        (
            "workflow:\n  mode: fixed\n  max_total_model_calls: 2\n  max_specialist_calls: 2\n",
            ">= 3",
        ),
        ("workflow:\n  max_unique_specialists: 4\n", "exceeds the number"),
        (
            "workflow:\n  available_specialists: [motion, motion]\n  fixed_specialists: []\n",
            "duplicate",
        ),
        (
            "workflow:\n  available_specialists: [Motion]\n  fixed_specialists: []\n",
            "invalid role IDs",
        ),
        ("workflow:\n  horizon_seconds: 4\n", "horizon_seconds"),
        ("alerts:\n  mode: dispatch\n", "alerts.mode"),
        ("alerts:\n  require_human_review: false\n", "require_human_review"),
        ("model:\n  provider: qwen_api\n", "model.provider"),
        ("model:\n  request_timeout_seconds: .inf\n", "request_timeout_seconds"),
        ("video:\n  window_seconds: .nan\n", "window_seconds"),
        ("logging:\n  save_private_frames: yes\n", "save_private_frames"),
        ("logging:\n  cache_mode: false\n", "cache_mode"),
        ("workflow:\n  mode: fixed\n  mode: adaptive\n", "duplicate key"),
        ("video:\n  ? [a, b]\n  : 1\n", "scalars"),
        ("workflow:\n  horizon_seconds: 5.0\n", "horizon_seconds"),
        ("alerts:\n  require_human_review: 1\n", "require_human_review"),
        ("schema_version: true\n", "schema_version"),
        ("video:\n  window_seconds: 1:30\n", "window_seconds"),
        ("video:\n  max_frames: 020\n", "max_frames"),
        ("video:\n  window_seconds: true\n", "window_seconds"),
        ("model:\n  temperature: true\n", "temperature"),
        ("video:\n  window_seconds: '5'\n", "window_seconds"),
        ("video:\n  reject_out_of_order: false\n", "reject_out_of_order"),
        ("workflow:\n  max_total_model_calls: 3\n", "max_total_model_calls >= 4"),
    ],
)
def test_rejects_unknown_or_impossible_settings(tmp_path: Path, body: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_config(_overlay(tmp_path, body))


def test_missing_key_is_an_error_not_a_hidden_default(tmp_path: Path) -> None:
    path = _write(tmp_path, "partial.yaml", "schema_version: 1\nvideo:\n  window_seconds: 5.0\n")
    with pytest.raises(ConfigError, match="model"):
        load_config(path)


def test_extends_cycle_and_bad_files(tmp_path: Path) -> None:
    _write(tmp_path, "a.yaml", "extends: b.yaml\n")
    _write(tmp_path, "b.yaml", "extends: a.yaml\n")
    with pytest.raises(ConfigError, match="cycle"):
        load_config(tmp_path / "a.yaml")
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(tmp_path / "missing.yaml")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(_write(tmp_path, "list.yaml", "- 1\n"))
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(_write(tmp_path, "bad.yaml", "video: [\n"))


def test_validate_config_round_trip(fake_cfg) -> None:
    assert validate_config(fake_cfg.model_dump(mode="json")) == fake_cfg


def test_numbers_and_ints_still_parse(tmp_path: Path) -> None:
    cfg = load_config(_overlay(tmp_path, "video:\n  window_seconds: 5\n  sample_fps: 2.5e0\n"))
    assert cfg.video.window_seconds == 5.0 and cfg.video.sample_fps == 2.5
    adaptive_c3 = load_config(
        _overlay(tmp_path, "workflow:\n  max_total_model_calls: 3\n  max_specialist_calls: 1\n")
    )
    assert (
        adaptive_c3.workflow.max_total_model_calls == 3
    )  # Spec C=3: route + one specialist + synthesis.


def test_non_utf8_config_is_a_config_error(tmp_path: Path) -> None:
    path = tmp_path / "latin1.yaml"
    path.write_bytes(b"schema_version: 1\nnote: \xff\xfe\n")
    with pytest.raises(ConfigError, match="UTF-8"):
        load_config(path)
