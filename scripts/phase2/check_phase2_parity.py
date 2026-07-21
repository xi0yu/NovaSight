"""Phase 2 parity table.

Loads every JSONL fixture under ``rust/fixtures/phase2/`` and prints a
compact parity table that shows which Python outputs are guaranteed
by the Rust contract tests (exact, byte-locked) and which are
best-effort (bounded, finite-checked).

Usage::

    python3 scripts/phase2/check_phase2_parity.py

The exit code is 0 when the schema is intact and every required key
is present; non-zero only when the schema or fixture files are
missing. Numeric values are sanity-checked, not compared.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "rust" / "fixtures" / "phase2"
SCHEMA_PATH = FIXTURE_DIR / "schema.json"

# Nested keys must be present in every record. ``EXACT`` keys must be
# present; ``BOUNDED`` keys are sanity-checked for finite values
# inside a plausibility window. Comparison against Python telemetry
# is left to the Rust contract tests; this script is the shape
# check.
EXACT = {
    "freshness-reset": (
        ("outcome", "admit"),
        ("outcome", "reason"),
    ),
    "static-target": (
        ("transform", "model_to_roi_scale_x"),
        ("transform", "model_to_roi_scale_y"),
    ),
    "dual-phase-control": (
        ("decision", "dx"),
        ("decision", "dy"),
        ("decision", "emit_allowed"),
        ("decision", "block_reason"),
    ),
    "target-switch-loss": (("expected_target", "lock_reason"),),
    "moving-target": (),
}

BOUNDED = {
    "freshness-reset": (("outcome", "age_ms"),),
    "static-target": (),
    "dual-phase-control": (
        ("decision", "velocity_x"),
        ("decision", "velocity_y"),
        ("decision", "atan_response_x"),
        ("decision", "atan_response_y"),
        ("decision", "filtered_error_meas_x"),
        ("decision", "filtered_error_meas_y"),
        ("decision", "predicted_offset_x"),
        ("decision", "predicted_offset_y"),
        ("decision", "quantizer_residual_x"),
        ("decision", "quantizer_residual_y"),
    ),
    "target-switch-loss": (),
    "moving-target": (),
}

# ``outcome.reason`` is `null` when the freshness gate admits; we
# treat that as present and only require the field to exist.
NULLABLE_EXACT = {("freshness-reset", ("outcome", "reason"))}


def _lookup(record: dict, path: tuple[str, ...]):
    node: object = record
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return _MISSING
        node = node[key]
    return node


_MISSING = object()


def _check_exact(name: str, record: dict) -> list[str]:
    issues: list[str] = []
    for path in EXACT.get(name, ()):
        value = _lookup(record, path)
        if value is _MISSING:
            issues.append(f"missing exact key {'/'.join(path)}")
            continue
        if value is None and (name, path) in NULLABLE_EXACT:
            continue
        if value is None:
            issues.append(f"null exact key {'/'.join(path)}")
    return issues


def _check_bounded(name: str, record: dict) -> list[str]:
    issues: list[str] = []
    for path in BOUNDED.get(name, ()):
        value = _lookup(record, path)
        if value is _MISSING or value is None:
            continue
        if not isinstance(value, (int, float)):
            issues.append(f"non-numeric {path}: {value!r}")
            continue
        if isinstance(value, float) and (
            value != value or value in (float("inf"), float("-inf"))
        ):
            issues.append(f"non-finite {path}: {value!r}")
            continue
        if abs(value) > 1e9:
            issues.append(f"{path}={value!r} out of plausibility window")
    return issues


def check_record(name: str, record: dict) -> list[str]:
    return [*_check_exact(name, record), *_check_bounded(name, record)]


def main() -> int:
    if not SCHEMA_PATH.exists():
        print(f"missing schema: {SCHEMA_PATH}", file=sys.stderr)
        return 1
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    if schema.get("version") != "1":
        print(f"unsupported schema version: {schema.get('version')}", file=sys.stderr)
        return 1

    print("Phase 2 parity table (shape check; numeric parity is in Rust contract tests)")
    print("=" * 80)
    summary: list[tuple[str, int, int, int, str]] = []
    for path in sorted(FIXTURE_DIR.glob("*.jsonl")):
        name = path.stem
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        issues: list[str] = []
        for record in records:
            issues.extend(check_record(name, record))
        exact = len(EXACT.get(name, ()))
        bounded = len(BOUNDED.get(name, ()))
        status = "OK" if not issues else f"FAIL ({len(issues)} issues)"
        summary.append((name, len(records), exact, bounded, status))
        for issue in issues:
            print(f"  {name}: {issue}")

    print("=" * 80)
    print(f"{'fixture':28s} {'records':>8s} {'exact':>6s} {'bounded':>8s} {'status'}")
    for name, count, exact, bounded, status in summary:
        print(f"{name:28s} {count:>8d} {exact:>6d} {bounded:>8d} {status}")

    if any("FAIL" in row[4] for row in summary):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
