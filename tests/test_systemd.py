from __future__ import annotations

import socket

from novasight.systemd import SystemdNotifier


class FakeNotifySocket:
    def __init__(self) -> None:
        self.timeout_s: float | None = None
        self.address = ""
        self.message = b""

    def __enter__(self) -> FakeNotifySocket:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def settimeout(self, timeout_s: float) -> None:
        self.timeout_s = timeout_s

    def connect(self, address: str) -> None:
        assert self.timeout_s is not None
        self.address = address

    def sendall(self, message: bytes) -> None:
        self.message = message


def test_systemd_notify_sets_io_timeout_before_connect(monkeypatch) -> None:
    fake = FakeNotifySocket()
    monkeypatch.setenv("NOTIFY_SOCKET", "/run/systemd/notify")
    monkeypatch.setattr(socket, "socket", lambda *_args: fake)

    assert SystemdNotifier().notify("READY=1") is True
    assert fake.timeout_s == 0.1
    assert fake.address == "/run/systemd/notify"
    assert fake.message == b"READY=1"
