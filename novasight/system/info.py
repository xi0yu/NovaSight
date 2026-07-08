from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Any


@dataclass(frozen=True)
class SystemInfo:
    platform: dict[str, str]
    jetson: dict[str, Any]
    gpu: dict[str, Any]
    tegrastats: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def collect_system_info() -> SystemInfo:
    return SystemInfo(
        platform={
            "system": platform.system(),
            "machine": platform.machine(),
            "release": platform.release(),
            "python": platform.python_version(),
        },
        jetson=_jetson_info(),
        gpu=_gpu_info(),
        tegrastats=_tegrastats_sample(),
    )


def _jetson_info() -> dict[str, Any]:
    nv_tegra = _read_first_existing(
        Path("/etc/nv_tegra_release"),
        Path("/etc/nvidia_tegra_release"),
    )
    jetpack = _read_first_existing(
        Path("/etc/nvidia-jetpack"),
        Path("/etc/jetpack-release"),
    )
    return {
        "is_jetson": bool(nv_tegra or jetpack or platform.machine() == "aarch64"),
        "nv_tegra_release": nv_tegra,
        "jetpack_release": jetpack,
    }


def _gpu_info() -> dict[str, Any]:
    temperatures = _gpu_temperatures()
    return {
        "temperature_c": temperatures[0]["temperature_c"] if temperatures else None,
        "thermal_zones": temperatures,
    }


def _gpu_temperatures() -> list[dict[str, Any]]:
    zones: list[dict[str, Any]] = []
    for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        zone_type = _read_text(zone / "type")
        if "gpu" not in zone_type.lower():
            continue
        temp_text = _read_text(zone / "temp")
        try:
            raw_temp = float(temp_text)
        except ValueError:
            continue
        temperature_c = raw_temp / 1000.0 if raw_temp > 200.0 else raw_temp
        zones.append(
            {
                "zone": zone.name,
                "type": zone_type,
                "temperature_c": round(temperature_c, 2),
            }
        )
    return zones


def _tegrastats_sample() -> dict[str, Any]:
    binary = shutil.which("tegrastats")
    if binary is None:
        return {"available": False, "sample": "", "reason": "tegrastats not found"}
    process = subprocess.Popen(
        [binary, "--interval", "100"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=0.35)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate(timeout=1.0)
    sample = next((line.strip() for line in stdout.splitlines() if line.strip()), "")
    return {
        "available": bool(sample),
        "sample": sample,
        "reason": "" if sample else (stderr.strip() or "tegrastats produced no sample"),
    }


def _read_first_existing(*paths: Path) -> str:
    for path in paths:
        text = _read_text(path)
        if text:
            return text
    return ""


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


__all__ = ["SystemInfo", "collect_system_info"]
