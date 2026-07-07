from __future__ import annotations

from novasight.config import RuntimeConfig

from .contracts import IHardwareBox
from .kmbox_net import KmboxNetAdapter


def create_hardware_box(config: RuntimeConfig) -> IHardwareBox | None:
    kind = config.hardware.kind.strip().lower()
    if kind != "kmnet":
        raise ValueError("hardware.kind must be kmnet")
    if not config.hardware.host.strip() or config.hardware.port <= 0:
        return None
    return KmboxNetAdapter(config.hardware.host, config.hardware.port)
