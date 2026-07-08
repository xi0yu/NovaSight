from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import platform
import re
import resource
import shutil
import subprocess
import time
from typing import Any

_PROCESS_START_MONOTONIC_S = time.monotonic()
_MEMORY_RE = re.compile(r"\b(?P<name>RAM|SWAP)\s+(?P<used>\d+)/(?P<total>\d+)MB")
_CPU_RE = re.compile(r"\bCPU\s+\[(?P<items>[^\]]*)\]")
_FREQ_RE = re.compile(r"\b(?P<name>GR3D_FREQ|EMC_FREQ)\s+(?P<load>\d+)%@(?P<freq>\d+)")
_TEMP_RE = re.compile(r"\b(?P<name>[A-Za-z0-9_]+)@(?P<temp>-?\d+(?:\.\d+)?)C")


@dataclass(frozen=True)
class SystemInfo:
    platform: dict[str, str]
    process: dict[str, Any]
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
        process=_process_info(),
        jetson=_jetson_info(),
        gpu=_gpu_info(),
        tegrastats=_tegrastats_sample(),
    )


def _process_info() -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss_bytes = _ru_maxrss_bytes(float(getattr(usage, "ru_maxrss", 0.0)))
    uptime_s = max(0.0, time.monotonic() - _PROCESS_START_MONOTONIC_S)
    return {
        "pid": os.getpid(),
        "uptime_s": round(uptime_s, 3),
        "rss_bytes": rss_bytes,
        "rss_mb": round(rss_bytes / (1024 * 1024), 3),
    }


def _ru_maxrss_bytes(value: float) -> int:
    if platform.system() == "Darwin":
        return max(0, int(value))
    return max(0, int(value * 1024))


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
    parsed = _parse_tegrastats_sample(sample)
    return {
        "available": bool(sample),
        "sample": sample,
        "reason": "" if sample else (stderr.strip() or "tegrastats produced no sample"),
        **parsed,
    }


def _parse_tegrastats_sample(sample: str) -> dict[str, Any]:
    if not sample:
        return {}
    parsed: dict[str, Any] = {}
    memory = _parse_tegrastats_memory(sample)
    if memory:
        parsed.update(memory)
    cpu = _parse_tegrastats_cpu(sample)
    if cpu:
        parsed["cpu"] = cpu
    frequencies = _parse_tegrastats_frequencies(sample)
    if frequencies:
        parsed.update(frequencies)
    temperatures = {
        match.group("name").lower(): float(match.group("temp"))
        for match in _TEMP_RE.finditer(sample)
    }
    if temperatures:
        parsed["temperatures_c"] = temperatures
    return parsed


def _parse_tegrastats_memory(sample: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for match in _MEMORY_RE.finditer(sample):
        name = match.group("name").lower()
        used = int(match.group("used"))
        total = int(match.group("total"))
        parsed[f"{name}_mb"] = {
            "used": used,
            "total": total,
            "pct": round((used / total) * 100.0, 2) if total > 0 else 0.0,
        }
    return parsed


def _parse_tegrastats_cpu(sample: str) -> list[dict[str, Any]]:
    match = _CPU_RE.search(sample)
    if match is None:
        return []
    cores: list[dict[str, Any]] = []
    for index, raw_item in enumerate(match.group("items").split(",")):
        item = raw_item.strip()
        if item == "off":
            cores.append({"core": index, "online": False, "load_pct": 0, "freq_mhz": 0})
            continue
        load_text, sep, freq_text = item.partition("%@")
        if not sep:
            continue
        try:
            cores.append(
                {
                    "core": index,
                    "online": True,
                    "load_pct": int(load_text),
                    "freq_mhz": int(freq_text),
                }
            )
        except ValueError:
            continue
    return cores


def _parse_tegrastats_frequencies(sample: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for match in _FREQ_RE.finditer(sample):
        key = match.group("name").lower()
        parsed[key] = {
            "load_pct": int(match.group("load")),
            "freq_mhz": int(match.group("freq")),
        }
    return parsed


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
