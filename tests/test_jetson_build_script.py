from __future__ import annotations

import os
import subprocess
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

