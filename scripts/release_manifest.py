#!/usr/bin/env python3
"""Generate and verify the immutable NovaSight release payload manifest."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any


MANIFEST_NAME = "SHA256SUMS.json"
MANIFEST_VERSION = 1
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    release_id: str
    files_checked: int
    missing: tuple[str, ...] = ()
    unexpected: tuple[str, ...] = ()
    mismatched: tuple[str, ...] = ()
    unsafe: tuple[str, ...] = ()
    error: str = ""


def generate_manifest(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    release_id = _read_release_id(root)
    files, unsafe = _inventory(root)
    if unsafe:
        raise ValueError(f"release contains unsafe paths: {', '.join(unsafe)}")
    return {
        "version": MANIFEST_VERSION,
        "release_id": release_id,
        "files": {relative: _sha256(path) for relative, path in files.items()},
    }


def write_manifest(root: Path) -> Path:
    root = root.resolve(strict=True)
    manifest = generate_manifest(root)
    destination = root / MANIFEST_NAME
    temporary = root / f".{MANIFEST_NAME}.{os.getpid()}.tmp"
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination


def verify_manifest(root: Path) -> VerificationResult:
    try:
        root = root.resolve(strict=True)
        release_id = _read_release_id(root)
        payload = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
        expected = _validated_files(payload, release_id)
        actual, unsafe = _inventory(root)
        expected_names = set(expected)
        actual_names = set(actual)
        missing = tuple(sorted(expected_names - actual_names))
        unexpected = tuple(sorted(actual_names - expected_names))
        mismatched = tuple(
            relative
            for relative in sorted(expected_names & actual_names)
            if _sha256(actual[relative]) != expected[relative]
        )
        return VerificationResult(
            passed=not (missing or unexpected or mismatched or unsafe),
            release_id=release_id,
            files_checked=len(expected_names & actual_names),
            missing=missing,
            unexpected=unexpected,
            mismatched=mismatched,
            unsafe=tuple(unsafe),
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return VerificationResult(
            passed=False,
            release_id="",
            files_checked=0,
            error=str(error),
        )


def _validated_files(payload: Any, release_id: str) -> dict[str, str]:
    if not isinstance(payload, dict) or payload.get("version") != MANIFEST_VERSION:
        raise ValueError(f"manifest version must be {MANIFEST_VERSION}")
    if payload.get("release_id") != release_id:
        raise ValueError("manifest release_id does not match RELEASE_ID")
    raw_files = payload.get("files")
    if not isinstance(raw_files, dict) or not raw_files:
        raise ValueError("manifest files must be a non-empty object")
    files: dict[str, str] = {}
    for raw_path, raw_digest in raw_files.items():
        if not isinstance(raw_path, str) or not _safe_relative_path(raw_path):
            raise ValueError(f"manifest contains unsafe path: {raw_path!r}")
        if raw_path == MANIFEST_NAME:
            raise ValueError("manifest cannot hash itself")
        if not isinstance(raw_digest, str) or _DIGEST.fullmatch(raw_digest) is None:
            raise ValueError(f"manifest contains invalid SHA-256 for {raw_path}")
        files[raw_path] = raw_digest
    return files


def _inventory(root: Path) -> tuple[dict[str, Path], list[str]]:
    files: dict[str, Path] = {}
    unsafe: list[str] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if _ignored(relative):
            continue
        if path.is_symlink():
            unsafe.append(relative)
        elif path.is_file():
            files[relative] = path
        elif not path.is_dir():
            unsafe.append(relative)
    return files, unsafe


def _ignored(relative: str) -> bool:
    path = PurePosixPath(relative)
    return (
        relative == MANIFEST_NAME
        or relative == ".complete"
        or "__pycache__" in path.parts
        or path.suffix in {".pyc", ".pyo"}
        or (len(path.parts) == 1 and relative.startswith(f".{MANIFEST_NAME}."))
    )


def _safe_relative_path(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts


def _read_release_id(root: Path) -> str:
    value = (root / "RELEASE_ID").read_text(encoding="utf-8").strip()
    if not value or len(value) > 64 or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) is None:
        raise ValueError(f"invalid RELEASE_ID: {value}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("generate", "verify"))
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    if args.operation == "generate":
        try:
            destination = write_manifest(args.root)
        except (OSError, ValueError) as error:
            print(f"release manifest generation failed: {error}", file=sys.stderr)
            return 2
        print(destination)
        return 0
    result = verify_manifest(args.root)
    print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))
    return 0 if result.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
