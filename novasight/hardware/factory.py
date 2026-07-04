from __future__ import annotations

from novasight.config import RuntimeConfig

from .contracts import IHardwareBox
from .kmbox_net import KmboxNetAdapter
from .makcu import MAKCUAdapter


def create_hardware_box(config: RuntimeConfig) -> IHardwareBox | None:
    kind = config.hardware.kind.strip().lower()
    if kind in {"", "none", "silent"}:
        return None
    if kind in {"kmnet", "kmbox", "kmbox_net"}:
        if not config.hardware.host.strip() or config.hardware.port <= 0:
            return None
        return KmboxNetAdapter(config.hardware.host, config.hardware.port)
    if kind == "makcu":
        if not config.hardware.serial_port.strip():
            raise ValueError("hardware.serial_port is required for MAKCU")
        return MAKCUAdapter(config.hardware.serial_port)
    raise ValueError(f"unknown hardware kind: {config.hardware.kind}")
