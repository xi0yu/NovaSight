from __future__ import annotations

import socket

from .contracts import BoxInputState


class KmboxNetAdapter:
    def __init__(
        self,
        host: str,
        port: int,
        *,
        timeout_s: float = 0.02,
        transport: str = "udp",
    ) -> None:
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.transport = transport
        self._socket: socket.socket | None = None
        self._last_input = BoxInputState()

    def connect(self) -> None:
        family = socket.SOCK_DGRAM if self.transport == "udp" else socket.SOCK_STREAM
        self._socket = socket.socket(socket.AF_INET, family)
        self._socket.settimeout(self.timeout_s)
        if self.transport != "udp":
            self._socket.connect((self.host, self.port))

    def send_move(self, dx: int, dy: int) -> None:
        self._send(f"move {dx} {dy}\n")

    def send_click(self, button: str) -> None:
        self._send(f"click {button}\n")

    def get_input_state(self) -> BoxInputState:
        sock = self._require_socket()
        try:
            data, _addr = sock.recvfrom(256)
        except TimeoutError:
            return self._last_input
        except socket.timeout:
            return self._last_input
        text = data.decode("utf-8", errors="ignore").strip().lower()
        state = BoxInputState(
            left="left=1" in text or "l=1" in text,
            right="right=1" in text or "r=1" in text,
            side="side=1" in text or "x=1" in text,
            raw={"packet": text},
        )
        self._last_input = state
        return state

    def _send(self, command: str) -> None:
        sock = self._require_socket()
        payload = command.encode("ascii")
        if self.transport == "udp":
            sock.sendto(payload, (self.host, self.port))
        else:
            sock.sendall(payload)

    def _require_socket(self) -> socket.socket:
        if self._socket is None:
            raise RuntimeError("hardware box is not connected")
        return self._socket
