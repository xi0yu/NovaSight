from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_jetson_preprocess.sh"


def test_jetson_preprocess_build_script_is_directly_runnable() -> None:
    assert SCRIPT.is_file()
    assert os.access(SCRIPT, os.X_OK)

    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)

    text = SCRIPT.read_text(encoding="utf-8")
    assert "doctor jetson-native-build" in text
    assert "libnovasight_preprocess.so" in text
    assert "NOVASIGHT_JETSON_NATIVE_LIBRARY" in text
    assert "--skip-load" in text
    assert "jetson-native-smoke" not in text


def test_jetson_preprocess_build_script_fails_fast_off_jetson(tmp_path: Path) -> None:
    if platform.system() == "Linux" and platform.machine() in {"aarch64", "arm64"}:
        return
    python_bin = ROOT / ".venv" / "bin" / "python3"
    if not python_bin.exists():
        python_bin = Path(sys.executable)

    result = subprocess.run(
        [
            str(SCRIPT),
            "--python",
            str(python_bin),
            "--build-dir",
            str(tmp_path / "jetson-native"),
            "--skip-load",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 2
    assert "reason: jetson_native_build_requires_jetson" in output
    assert "configure_command:" not in output
