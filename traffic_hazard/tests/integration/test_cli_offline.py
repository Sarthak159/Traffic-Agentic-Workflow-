from __future__ import annotations

import socket

import pytest
from traffic_hazard.cli import main

from tests.conftest import CONFIGS, NetworkAccessError


def test_doctor_reports_presence_without_secret_values(monkeypatch, capsys) -> None:
    monkeypatch.setenv("QWEN_API_KEY", "sk-very-secret-value")
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "sk-very-secret-value" not in out
    assert "QWEN_API_KEY   set" in out
    assert "disabled (no-op)" in out


def test_config_check_exit_codes(tmp_path, capsys) -> None:
    assert main(["config-check", "--config", str(CONFIGS / "local_fake.yaml")]) == 0
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema_version: 2\n")
    assert main(["config-check", "--config", str(bad)]) == 2
    assert "config error" in capsys.readouterr().err


def test_self_test_passes_offline(capsys) -> None:
    assert main(["self-test", "--config", str(CONFIGS / "local_fake.yaml")]) == 0
    assert "self-test passed" in capsys.readouterr().out


def test_network_guard_is_active() -> None:
    with pytest.raises(NetworkAccessError):
        socket.create_connection(("example.com", 443), timeout=1)
    with pytest.raises(NetworkAccessError):
        socket.getaddrinfo("api.example.com", 443)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(NetworkAccessError):
            sock.connect(("127.0.0.1", 9))
        with pytest.raises(NetworkAccessError):
            sock.connect_ex(("127.0.0.1", 9))
    finally:
        sock.close()


def test_self_test_default_config_is_cwd_independent(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["self-test"]) == 0
    assert "self-test passed" in capsys.readouterr().out
