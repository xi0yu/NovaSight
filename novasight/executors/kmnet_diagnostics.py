from __future__ import annotations

import socket
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class KmNetRouteStatus:
    available: bool
    resolved_ip: str = ""
    local_ip: str = ""
    error: str = ""

    def asdict(self) -> dict[str, object]:
        return asdict(self)


def probe_kmnet_route(host: str, port: int) -> KmNetRouteStatus:
    try:
        addresses = socket.getaddrinfo(
            host,
            int(port),
            family=socket.AF_INET,
            type=socket.SOCK_DGRAM,
        )
        if not addresses:
            return KmNetRouteStatus(available=False, error="host did not resolve to an IPv4 address")
        resolved_ip = str(addresses[0][4][0])
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route_socket:
            route_socket.connect((resolved_ip, int(port)))
            local_ip = str(route_socket.getsockname()[0])
    except OSError as exc:
        return KmNetRouteStatus(available=False, error=f"{type(exc).__name__}: {exc}")
    return KmNetRouteStatus(
        available=True,
        resolved_ip=resolved_ip,
        local_ip=local_ip,
    )


__all__ = ["KmNetRouteStatus", "probe_kmnet_route"]
