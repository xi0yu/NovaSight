from __future__ import annotations

import json
from pathlib import Path

from scripts.release_manifest import generate_manifest, verify_manifest, write_manifest


def _release(tmp_path: Path) -> Path:
    root = tmp_path / "release"
    (root / "bin").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "RELEASE_ID").write_text("abc123\n", encoding="utf-8")
    (root / "bin" / "novasightd").write_bytes(b"daemon")
    (root / "scripts" / "helper.py").write_text("print('ready')\n", encoding="utf-8")
    return root


def test_manifest_round_trip_covers_every_payload_file(tmp_path: Path) -> None:
    root = _release(tmp_path)

    destination = write_manifest(root)
    result = verify_manifest(root)

    assert destination == root / "SHA256SUMS.json"
    assert result.passed
    assert result.release_id == "abc123"
    assert result.files_checked == 3


def test_manifest_rejects_changed_and_unlisted_payloads(tmp_path: Path) -> None:
    root = _release(tmp_path)
    write_manifest(root)
    (root / "bin" / "novasightd").write_bytes(b"replaced")
    (root / "unexpected.txt").write_text("not staged", encoding="utf-8")

    result = verify_manifest(root)

    assert not result.passed
    assert result.mismatched == ("bin/novasightd",)
    assert result.unexpected == ("unexpected.txt",)


def test_manifest_rejects_release_id_relabeling_and_unsafe_paths(tmp_path: Path) -> None:
    root = _release(tmp_path)
    manifest = generate_manifest(root)
    manifest["files"]["../outside"] = "0" * 64
    (root / "SHA256SUMS.json").write_text(json.dumps(manifest), encoding="utf-8")

    unsafe = verify_manifest(root)
    assert not unsafe.passed
    assert "unsafe path" in unsafe.error

    write_manifest(root)
    (root / "RELEASE_ID").write_text("different\n", encoding="utf-8")
    relabeled = verify_manifest(root)
    assert not relabeled.passed
    assert "does not match" in relabeled.error


def test_manifest_ignores_installer_completion_and_python_cache(tmp_path: Path) -> None:
    root = _release(tmp_path)
    write_manifest(root)
    (root / ".complete").touch()
    cache = root / "scripts" / "__pycache__"
    cache.mkdir()
    (cache / "helper.cpython-310.pyc").write_bytes(b"cache")

    assert verify_manifest(root).passed
