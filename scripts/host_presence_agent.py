#!/usr/bin/env python3
"""Send target-host presence heartbeats to a NovaSight Jetson runtime."""

from __future__ import annotations

import argparse
import json
import platform
import socket
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def heartbeat_url(base_url: str) -> str:
    return f"{str(base_url).rstrip('/')}/api/runtime/presence/heartbeat"


def send_heartbeat(base_url: str, host_id: str, *, timeout_s: float = 3.0) -> dict:
    payload = json.dumps({"host_id": str(host_id)}, ensure_ascii=False).encode("utf-8")
    request = Request(
        heartbeat_url(base_url),
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=max(0.1, float(timeout_s))) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Keep NovaSight inference active while this target host is online."
    )
    parser.add_argument(
        "--jetson",
        required=True,
        help="Jetson URL, for example http://192.168.2.10:5174",
    )
    parser.add_argument(
        "--host-id",
        default=socket.gethostname() or platform.node() or "target-host",
        help="Stable target host identifier",
    )
    parser.add_argument("--interval", type=float, default=2.0, help="Heartbeat interval in seconds")
    parser.add_argument("--timeout", type=float, default=3.0, help="HTTP timeout in seconds")
    args = parser.parse_args()

    last_error = ""
    last_mode = ""
    while True:
        try:
            response = send_heartbeat(args.jetson, args.host_id, timeout_s=args.timeout)
            mode = str(dict(response.get("power_saving") or {}).get("mode", "unknown"))
            if last_error or mode != last_mode:
                print(f"NovaSight heartbeat ok host={args.host_id} mode={mode}", flush=True)
            last_error = ""
            last_mode = mode
        except (HTTPError, URLError, OSError, ValueError, json.JSONDecodeError) as exc:
            message = str(exc)
            if message != last_error:
                print(f"NovaSight heartbeat failed: {message}", flush=True)
                last_error = message
        time.sleep(max(0.5, float(args.interval)))


if __name__ == "__main__":
    raise SystemExit(main())
