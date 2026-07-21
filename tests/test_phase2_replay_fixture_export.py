"""Regression tests for the Phase 2 deterministic fixture exporter.

Running the exporter twice must produce byte-identical output. The exporter
must reject NaN / Inf and accept only the schema version declared in
``rust/fixtures/phase2/schema.json``. A separate test pins the exporter's
CLI so that the canonical Phase 2 fixture set on disk is regenerated from
the same source tree.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "rust" / "fixtures" / "phase2"
SCHEMA_PATH = FIXTURE_DIR / "schema.json"
EXPORTER = REPO_ROOT / "scripts" / "phase2" / "export_replay_fixtures.py"

EXPECTED_FILES = (
    "static-target.jsonl",
    "moving-target.jsonl",
    "freshness-reset.jsonl",
    "dual-phase-control.jsonl",
    "target-switch-loss.jsonl",
)


def test_schema_version_is_pinned() -> None:
    assert SCHEMA_PATH.exists(), f"missing {SCHEMA_PATH}"
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["version"] == "1"
    assert schema["epoch_unit"] == "u64"
    assert schema["generation_unit"] == "u64"
    assert schema["monotonic_time_unit"] == "ns"


@pytest.mark.skipif(not EXPORTER.exists(), reason="exporter not yet written")
def test_exporter_runs_and_is_deterministic() -> None:
    captured = {}
    for filename in EXPECTED_FILES:
        path = FIXTURE_DIR / filename
        assert path.exists(), f"missing fixture {filename}; run the exporter"
        captured[filename] = path.read_bytes()

    result = subprocess.run(
        [sys.executable, str(EXPORTER)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "phase2 fixtures regenerated" in result.stdout

    for filename in EXPECTED_FILES:
        path = FIXTURE_DIR / filename
        assert path.read_bytes() == captured[filename], (
            f"{filename} is not deterministic across exporter runs"
        )


def test_no_nan_or_inf_in_fixtures() -> None:
    for filename in EXPECTED_FILES:
        path = FIXTURE_DIR / filename
        if not path.exists():
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            data = json.loads(line)
            _assert_finite(data, path, line_number)


def _assert_finite(node, path: Path, line_number: int) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, float) and not math.isfinite(value):
                pytest.fail(f"{path.name}:{line_number} non-finite {key}={value!r}")
            _assert_finite(value, path, line_number)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _assert_finite(value, path, line_number)


def test_freshness_outcomes_cover_required_branches() -> None:
    path = FIXTURE_DIR / "freshness-reset.jsonl"
    assert path.exists(), "run the exporter first"
    labels = {
        json.loads(line)["label"]
        for line in path.read_text(encoding="utf-8").splitlines()
    }
    assert {
        "admit_fresh",
        "exactly_at_limit",
        "one_ns_over",
        "clock_regression",
        "duplicate_generation",
        "strictest_inputs_empty",
    }.issubset(labels)


def test_dual_phase_control_records_capture_decision_shell() -> None:
    path = FIXTURE_DIR / "dual-phase-control.jsonl"
    assert path.exists(), "run the exporter first"
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        assert "observation" in record
        assert "decision" in record
        decision = record["decision"]
        assert set(decision) == {"dx", "dy", "emit_allowed", "block_reason", "telemetry_keys"}
        assert isinstance(decision["dx"], int)
        assert isinstance(decision["dy"], int)
        assert isinstance(decision["emit_allowed"], bool)
        assert isinstance(decision["telemetry_keys"], list)
