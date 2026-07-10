from __future__ import annotations

import socket

from novasight.executors.kmnet_diagnostics import probe_kmnet_route


class FakeRouteSocket:
    def __enter__(self) -> FakeRouteSocket:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def connect(self, _address: tuple[str, int]) -> None:
        return None

    def getsockname(self) -> tuple[str, int]:
        return ("192.168.2.10", 45678)


def test_kmnet_route_reports_selected_local_address(monkeypatch) -> None:
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [(socket.AF_INET, socket.SOCK_DGRAM, 0, "", ("192.168.2.188", 8888))],
    )
    monkeypatch.setattr(socket, "socket", lambda *_args: FakeRouteSocket())

    status = probe_kmnet_route("127.0.0.1", 8888)

    assert status.available is True
    assert status.resolved_ip == "192.168.2.188"
    assert status.local_ip == "192.168.2.10"
    assert status.error == ""


def test_invalid_kmnet_host_reports_resolution_error(monkeypatch) -> None:
    def fail_resolution(*_args, **_kwargs):
        raise socket.gaierror("not found")

    monkeypatch.setattr(socket, "getaddrinfo", fail_resolution)

    status = probe_kmnet_route("invalid host name", 8888)

    assert status.available is False
    assert status.error
