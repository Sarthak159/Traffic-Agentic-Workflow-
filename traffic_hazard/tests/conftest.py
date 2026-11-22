"""Shared test setup. Offline tests must never open an internet socket."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = APP_ROOT / "configs"


class NetworkAccessError(RuntimeError):
    pass


@pytest.fixture(autouse=True)
def _block_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("live"):
        return
    real_connect = socket.socket.connect

    def guarded_connect(self: socket.socket, address: object) -> None:
        if self.family in (socket.AF_INET, socket.AF_INET6):
            raise NetworkAccessError(f"offline test attempted a network connection to {address!r}")
        real_connect(self, address)  # type: ignore[arg-type]

    def guarded_create_connection(
        address: object, *args: object, **kwargs: object
    ) -> socket.socket:
        raise NetworkAccessError(f"offline test attempted a network connection to {address!r}")

    real_connect_ex = socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo

    def guarded_connect_ex(self: socket.socket, address: object) -> int:
        if self.family in (socket.AF_INET, socket.AF_INET6):
            raise NetworkAccessError(f"offline test attempted a network connection to {address!r}")
        return real_connect_ex(self, address)  # type: ignore[arg-type]

    def guarded_getaddrinfo(host: object, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        # Name resolution of a remote host would already leak a DNS query.
        if host not in (None, "localhost", "127.0.0.1", "::1", b"localhost"):
            raise NetworkAccessError(f"offline test attempted DNS resolution of {host!r}")
        return real_getaddrinfo(host, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "create_connection", guarded_create_connection)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)


@pytest.fixture(autouse=True)
def _tracing_disabled() -> None:
    from agents import set_tracing_disabled

    set_tracing_disabled(True)


@pytest.fixture
def fake_cfg():
    from traffic_hazard.config import load_config

    return load_config(CONFIGS / "local_fake.yaml")
