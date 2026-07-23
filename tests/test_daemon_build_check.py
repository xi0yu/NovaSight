from __future__ import annotations

import json
from pathlib import Path
import stat

from scripts.daemon_build_check import check_daemon_build


def _daemon(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "novasightd"
    encoded = json.dumps(payload, separators=(",", ":"))
    path.write_text(f"#!/bin/sh\nprintf '%s\\n' '{encoded}'\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _production_info() -> dict[str, object]:
    return {
        "schema_version": 1,
        "binary": "novasightd",
        "version": "0.1.0",
        "source_revision": "a" * 40,
        "source_dirty": False,
        "target": "aarch64-unknown-linux-gnu",
        "profile": "release",
        "features": ["deepstream"],
    }


def test_accepts_matching_aarch64_deepstream_release(tmp_path: Path) -> None:
    info = _production_info()
    check = check_daemon_build(
        _daemon(tmp_path, info),
        expected_revision="a" * 40,
        expected_dirty=False,
    )

    assert check.passed
    assert check.errors == ()
    assert check.build_info == info


def test_rejects_stale_or_non_production_daemon(tmp_path: Path) -> None:
    info = _production_info()
    info.update(
        source_revision="b" * 40,
        source_dirty=True,
        target="x86_64-apple-darwin",
        profile="debug",
        features=[],
    )

    check = check_daemon_build(
        _daemon(tmp_path, info),
        expected_revision="a" * 40,
        expected_dirty=False,
    )

    assert not check.passed
    assert any("source_revision" in error for error in check.errors)
    assert any("source_dirty" in error for error in check.errors)
    assert any("aarch64 Linux" in error for error in check.errors)
    assert any("profile must be release" in error for error in check.errors)
    assert any("deepstream feature" in error for error in check.errors)


def test_rejects_non_json_and_unknown_revision(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid"
    invalid.write_text("#!/bin/sh\necho not-json\n", encoding="utf-8")
    invalid.chmod(invalid.stat().st_mode | stat.S_IXUSR)
    assert not check_daemon_build(invalid).passed

    info = _production_info()
    info["source_revision"] = "unknown"
    unknown = check_daemon_build(_daemon(tmp_path, info))
    assert not unknown.passed
    assert "source_revision is unavailable" in unknown.errors
