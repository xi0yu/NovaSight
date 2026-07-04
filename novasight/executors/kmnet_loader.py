from __future__ import annotations

import importlib
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class KmNetLoadResult:
    module: Any | None
    available: bool
    source: str
    reason: str
    platform: str
    machine: str
    python_tag: str


def load_kmnet_driver() -> KmNetLoadResult:
    platform_name = sys.platform
    machine = platform.machine().lower()
    python_tag = f"cpython-{sys.version_info.major}{sys.version_info.minor}"

    try:
        module = importlib.import_module("kmNet")
        return KmNetLoadResult(
            module=module,
            available=True,
            source="python-path",
            reason="",
            platform=platform_name,
            machine=machine,
            python_tag=python_tag,
        )
    except Exception as direct_exc:
        vendor_dir = _vendor_dir(platform_name, machine)
        if vendor_dir is None:
            return KmNetLoadResult(
                module=None,
                available=False,
                source="unsupported-platform",
                reason=(
                    f"kmNet driver unavailable: unsupported platform "
                    f"{platform_name}/{machine}; direct import failed: {direct_exc}"
                ),
                platform=platform_name,
                machine=machine,
                python_tag=python_tag,
            )
        if not vendor_dir.exists():
            return KmNetLoadResult(
                module=None,
                available=False,
                source=str(vendor_dir),
                reason=f"kmNet vendor directory is missing: {vendor_dir}",
                platform=platform_name,
                machine=machine,
                python_tag=python_tag,
            )

        if str(vendor_dir) not in sys.path:
            sys.path.insert(0, str(vendor_dir))
        importlib.invalidate_caches()
        try:
            module = importlib.import_module("kmNet")
        except Exception as vendor_exc:
            return KmNetLoadResult(
                module=None,
                available=False,
                source=str(vendor_dir),
                reason=(
                    f"kmNet vendor load failed from {vendor_dir}: {vendor_exc}; "
                    f"platform={platform_name}/{machine}, python={python_tag}. "
                    "Jetson vendor driver currently targets aarch64 + Python 3.10."
                ),
                platform=platform_name,
                machine=machine,
                python_tag=python_tag,
            )
        return KmNetLoadResult(
            module=module,
            available=True,
            source=str(vendor_dir),
            reason="",
            platform=platform_name,
            machine=machine,
            python_tag=python_tag,
        )


def _vendor_dir(platform_name: str, machine: str) -> Path | None:
    root = Path(__file__).resolve().parents[1] / "vendor" / "kmnet"
    if platform_name == "linux" and machine == "aarch64":
        return root / "linux"
    if platform_name == "darwin" and machine in {"arm64", "aarch64"}:
        return root / "mac"
    if platform_name == "win32":
        return root / "win"
    return None
