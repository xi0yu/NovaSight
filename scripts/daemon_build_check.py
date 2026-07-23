#!/usr/bin/env python3
"""Verify that a staged novasightd is the intended production build."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


@dataclass(frozen=True)
class DaemonBuildCheck:
    passed: bool
    build_info: dict[str, Any] | None
    errors: tuple[str, ...]


def check_daemon_build(
    daemon: Path,
    *,
    expected_revision: str | None = None,
    expected_dirty: bool | None = None,
) -> DaemonBuildCheck:
    try:
        result = subprocess.run(
            [str(daemon), "--build-info-json"],
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return DaemonBuildCheck(False, None, (f"cannot execute daemon: {error}",))
    try:
        info = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        return DaemonBuildCheck(False, None, (f"invalid build-info JSON: {error}",))
    if not isinstance(info, dict):
        return DaemonBuildCheck(False, None, ("build-info response is not an object",))
    errors: list[str] = []
    if result.returncode != 0:
        errors.append(f"build-info command exited {result.returncode}: {result.stderr.strip()}")
    if info.get("schema_version") != 1:
        errors.append("build-info schema_version must be 1")
    if info.get("binary") != "novasightd":
        errors.append("build-info binary must be novasightd")
    revision = info.get("source_revision")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdefABCDEF" for character in revision)
    ):
        errors.append("source_revision is unavailable")
    if expected_revision is not None and revision != expected_revision:
        errors.append(f"source_revision {revision!r} does not match {expected_revision!r}")
    dirty = info.get("source_dirty")
    if not isinstance(dirty, bool):
        errors.append("source_dirty must be boolean")
    if expected_dirty is not None and dirty != expected_dirty:
        errors.append(f"source_dirty {dirty!r} does not match {expected_dirty!r}")
    target = info.get("target")
    if not isinstance(target, str) or not target.startswith("aarch64-") or "linux" not in target:
        errors.append(f"target {target!r} is not aarch64 Linux")
    if info.get("profile") != "release":
        errors.append("profile must be release")
    features = info.get("features")
    if not isinstance(features, list) or "deepstream" not in features:
        errors.append("deepstream feature is not compiled into novasightd")
    return DaemonBuildCheck(not errors, info, tuple(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daemon", type=Path, required=True)
    parser.add_argument("--expected-revision")
    parser.add_argument("--expected-dirty", choices=("true", "false"))
    args = parser.parse_args()
    expected_dirty = None if args.expected_dirty is None else args.expected_dirty == "true"
    check = check_daemon_build(
        args.daemon,
        expected_revision=args.expected_revision,
        expected_dirty=expected_dirty,
    )
    print(json.dumps(asdict(check), ensure_ascii=False, sort_keys=True))
    return 0 if check.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
