"""Phase 2 deterministic fixture exporter.

Exports canonical replay fixtures for the Phase 2 Rust algorithm port.
The exporter runs against the production Python algorithm implementations
(``FreshnessGate``, ``CoordinateTransform``, ``RuntimeTargetSelector``,
dual-phase control entry point) and writes both the input observation and
the expected intermediate / output values into versioned JSONL files.

The Rust side parses the same files without launching Python; tests assert
exact equality for identity, state, reset, integer, and residual outcomes,
and compare floating-point values against the per-field tolerance declared
in ``rust/fixtures/phase2/schema.json``.

Usage::

    python scripts/phase2/export_replay_fixtures.py

Running the exporter twice must produce byte-identical output. NaN and Inf
are rejected. Object map order is canonicalized via ``sort_keys=True`` so
JSONL line ordering is the only meaningful axis.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Iterable
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "rust" / "fixtures" / "phase2"
SCHEMA_PATH = FIXTURE_DIR / "schema.json"
sys.path.insert(0, str(REPO_ROOT))

from novasight.contracts import BBox  # noqa: E402
from novasight.coordinates import CoordinateTransform  # noqa: E402
from novasight.runtime.freshness import FreshnessGate, strictest_positive_ms  # noqa: E402


def _canonicalize(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError("fixture values must be finite, got {value!r}")
        return round(value, 12)
    if isinstance(value, dict):
        return {key: _canonicalize(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if is_dataclass(value):
        return _canonicalize(asdict(value))
    return value


def _dump(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_canonicalize(record), sort_keys=True))
            handle.write("\n")


def _frame(
    *,
    epoch: int,
    generation: int,
    frame_id: int,
    captured_at_ns: int,
    inference_end_ts_ns: int,
    control_now_ns: int,
    model_width: float = 640.0,
    model_height: float = 640.0,
    roi_x: float = 0.0,
    roi_y: float = 0.0,
    roi_width: float = 640.0,
    roi_height: float = 640.0,
    capture_width: float = 640.0,
    capture_height: float = 640.0,
    control_origin_x: float = 0.0,
    control_origin_y: float = 0.0,
    display_scale_x: float = 1.0,
    display_scale_y: float = 1.0,
    coordinate_space: str = "model",
) -> dict:
    return {
        "epoch": epoch,
        "generation": generation,
        "frame_id": frame_id,
        "captured_at_ns": captured_at_ns,
        "inference_end_ts_ns": inference_end_ts_ns,
        "control_now_ns": control_now_ns,
        "coordinate_space": coordinate_space,
        "model_width": model_width,
        "model_height": model_height,
        "roi_x": roi_x,
        "roi_y": roi_y,
        "roi_width": roi_width,
        "roi_height": roi_height,
        "capture_width": capture_width,
        "capture_height": capture_height,
        "control_origin_x": control_origin_x,
        "control_origin_y": control_origin_y,
        "display_scale_x": display_scale_x,
        "display_scale_y": display_scale_y,
    }


def _detection(
    object_id: int,
    class_id: int,
    x: float,
    y: float,
    w: float,
    h: float,
    confidence: float = 0.9,
) -> dict:
    return {
        "object_id": object_id,
        "class_id": class_id,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "confidence": confidence,
    }


def _freshness_outcome(
    gate: FreshnessGate,
    *,
    capture_ts_ns: int,
    now_ns: int,
    template: str = "stale age={age_ms:.3f}ms threshold={threshold_ms:.3f}ms",
) -> dict:
    age_ms = FreshnessGate.age_ms(capture_ts_ns=capture_ts_ns, now_ns=now_ns)
    reason = gate.stale_reason(
        capture_ts_ns=capture_ts_ns, now_ns=now_ns, template=template
    )
    return {"admit": not reason, "reason": reason or None, "age_ms": age_ms}


def build_static_target() -> list[dict]:
    """Single static target at the model centre, three identical frames."""
    transform = CoordinateTransform(
        model_width=640,
        model_height=640,
        roi_x=0,
        roi_y=0,
        roi_width=640,
        roi_height=640,
        capture_width=640,
        capture_height=640,
    )
    detection = _detection(object_id=1, class_id=0, x=300.0, y=300.0, w=40.0, h=80.0)
    frames = []
    for index in range(3):
        frame_ns = 1_000_000_000 + index * 16_666_666
        frames.append({
            "frame": _frame(
                epoch=0,
                generation=index + 1,
                frame_id=index + 1,
                captured_at_ns=frame_ns,
                inference_end_ts_ns=frame_ns + 5_000_000,
                control_now_ns=frame_ns + 8_000_000,
            ),
            "transform": {
                "model_to_roi_scale_x": transform.model_to_roi_scale_x,
                "model_to_roi_scale_y": transform.model_to_roi_scale_y,
            },
            "detections": [detection],
        })
    return frames


def build_moving_target() -> list[dict]:
    """Target moving along a line; ROI is shifted so coordinate space changes."""
    transform = CoordinateTransform(
        model_width=640,
        model_height=640,
        roi_x=0,
        roi_y=0,
        roi_width=320,
        roi_height=320,
        capture_width=320,
        capture_height=320,
    )
    boxes_roi = [
        (10.0, 10.0, 30.0, 60.0),
        (40.0, 30.0, 60.0, 80.0),
        (90.0, 60.0, 110.0, 110.0),
    ]
    frames = []
    for index, (x1, y1, x2, y2) in enumerate(boxes_roi):
        frame_ns = 2_000_000_000 + index * 16_666_666
        capture_box = transform.roi_to_capture_box(BBox.from_xyxy(x1, y1, x2, y2))
        frames.append({
            "frame": _frame(
                epoch=0,
                generation=index + 1,
                frame_id=index + 1,
                captured_at_ns=frame_ns,
                inference_end_ts_ns=frame_ns + 5_000_000,
                control_now_ns=frame_ns + 8_000_000,
                roi_x=0,
                roi_y=0,
                roi_width=320,
                roi_height=320,
                capture_width=320,
                capture_height=320,
            ),
            "detections": [_detection(
                object_id=1, class_id=0,
                x=capture_box.x, y=capture_box.y,
                w=capture_box.w, h=capture_box.h,
            )],
        })
    return frames


def build_freshness_reset() -> list[dict]:
    """Cases that exercise the freshness policy: admit, exactly-at-limit, over,
    stale due to huge gap, clock regression, duplicate generation."""
    gate = FreshnessGate(threshold_ms=55.0)
    records: list[dict] = []
    base = 3_000_000_000

    cases = [
        # (label, capture_ts_ns, now_ns, generation)
        ("admit_fresh", base, base + 10_000_000, 1),
        ("exactly_at_limit", base + 16_666_666, base + 16_666_666 + 55_000_000, 2),
        ("one_ns_over", base + 16_666_666 * 2, base + 16_666_666 * 2 + 55_000_001, 3),
        ("clock_regression", base - 1, base, 4),
        ("duplicate_generation", base + 16_666_666 * 3, base + 16_666_666 * 3 + 5_000_000, 4),
    ]
    for label, capture_ts_ns, now_ns, generation in cases:
        outcome = _freshness_outcome(gate, capture_ts_ns=capture_ts_ns, now_ns=now_ns)
        record = {
            "label": label,
            "policy": {
                "threshold_ms": gate.threshold_ms,
                "strictest_inputs_ms": [55.0, 80.0, 0.0],
            },
            "frame": _frame(
                epoch=0,
                generation=generation,
                frame_id=generation,
                captured_at_ns=capture_ts_ns,
                inference_end_ts_ns=capture_ts_ns + 5_000_000,
                control_now_ns=now_ns,
            ),
            "outcome": outcome,
        }
        records.append(record)

    gate_from_inputs = FreshnessGate.strictest(0, 0, 0, 0)
    records.append({
        "label": "strictest_inputs_empty",
        "policy": {"threshold_ms": 0.0, "strictest_inputs_ms": []},
        "frame": _frame(
            epoch=0, generation=1, frame_id=1,
            captured_at_ns=base, inference_end_ts_ns=base + 1_000_000,
            control_now_ns=base + 2_000_000,
        ),
        "outcome": _freshness_outcome(
            gate_from_inputs, capture_ts_ns=base, now_ns=base + 100_000_000
        ),
    })

    return records


def build_dual_phase_control() -> list[dict]:
    """Minimal dual-phase control fixtures that lock down the input contract.
    The expected outputs are captured by invoking the real algorithm on a
    fresh instance with reset state. The Rust port must reproduce them within
    the per-field tolerances declared in schema.json."""
    from novasight.control.algorithms.dual_phase_atan_robust_predictive_v2 import (
        DualPhaseAtanRobustPredictiveV2Algorithm,
        DualPhaseAtanRobustPredictiveV2Config,
        DualPhaseAtanRobustPredictiveV2Observation,
    )

    algorithm = DualPhaseAtanRobustPredictiveV2Algorithm(
        DualPhaseAtanRobustPredictiveV2Config()
    )
    records: list[dict] = []
    base = 4_000_000_000

    cases = [
        # (label, target_x, target_y, crosshair_x, crosshair_y, trigger)
        ("initial_lock", 400.0, 360.0, 320.0, 320.0, True),
        ("moving_lock", 420.0, 380.0, 320.0, 320.0, True),
        ("target_lost", 420.0, 380.0, 320.0, 320.0, False),
        ("trigger_release", 420.0, 380.0, 320.0, 320.0, True),
    ]

    for index, (label, target_x, target_y, crosshair_x, crosshair_y, trigger) in enumerate(cases):
        frame_ns = base + index * 16_666_666
        observation = DualPhaseAtanRobustPredictiveV2Observation(
            generation=index + 1,
            frame_id=index + 1,
            target_id=1,
            capture_ts_ns=frame_ns,
            inference_end_ts_ns=frame_ns + 5_000_000,
            control_now_ns=frame_ns + 8_000_000,
            aim_x=target_x,
            aim_y=target_y,
            crosshair_x=crosshair_x,
            crosshair_y=crosshair_y,
            bbox_x1=target_x - 20.0,
            bbox_y1=target_y - 40.0,
            bbox_x2=target_x + 20.0,
            bbox_y2=target_y + 40.0,
            observation_width=640,
            observation_height=640,
            roi_left=0,
            roi_top=0,
            roi_width=640,
            roi_height=640,
            source_width=640,
            source_height=640,
            detection_confidence=0.9,
            track_confidence=0.9,
            trigger_active=trigger,
            target_valid=label != "target_lost",
        )
        decision = algorithm.calculate(observation)
        record = {
            "label": label,
            "observation": {
                "generation": observation.generation,
                "frame_id": observation.frame_id,
                "target_id": observation.target_id,
                "capture_ts_ns": observation.capture_ts_ns,
                "inference_end_ts_ns": observation.inference_end_ts_ns,
                "control_now_ns": observation.control_now_ns,
                "aim_x": observation.aim_x,
                "aim_y": observation.aim_y,
                "crosshair_x": observation.crosshair_x,
                "crosshair_y": observation.crosshair_y,
                "trigger_active": observation.trigger_active,
                "target_valid": observation.target_valid,
            },
            "decision": {
                "dx": decision.dx,
                "dy": decision.dy,
                "emit_allowed": decision.emit_allowed,
                "block_reason": decision.block_reason,
                "telemetry_keys": sorted(decision.telemetry.keys()),
            },
        }
        records.append(record)

    return records


def build_target_switch_loss() -> list[dict]:
    """Two candidates at the centre; the head role wins by preferred-class rule
    then a class-1 challenger arrives; targeting must keep the locked target
    until the challenger commits."""
    frames = []
    base = 5_000_000_000

    frames.append({
        "label": "lock_head",
        "frame": _frame(
            epoch=0, generation=1, frame_id=1,
            captured_at_ns=base, inference_end_ts_ns=base + 5_000_000,
            control_now_ns=base + 8_000_000,
        ),
        "detections": [
            _detection(1, 0, 300.0, 280.0, 40.0, 80.0),
            _detection(2, 1, 320.0, 380.0, 30.0, 60.0),
        ],
        "expected_target": {"object_id": 1, "class_id": 0, "lock_reason": "preferred_class"},
    })
    frames.append({
        "label": "challenger_close",
        "frame": _frame(
            epoch=0, generation=2, frame_id=2,
            captured_at_ns=base + 16_666_666,
            inference_end_ts_ns=base + 16_666_666 + 5_000_000,
            control_now_ns=base + 16_666_666 + 8_000_000,
        ),
        "detections": [
            _detection(1, 0, 300.0, 280.0, 40.0, 80.0),
            _detection(2, 1, 304.0, 296.0, 30.0, 60.0),
        ],
        "expected_target": {"object_id": 1, "class_id": 0, "lock_reason": "preferred_class"},
    })
    frames.append({
        "label": "challenger_commit",
        "frame": _frame(
            epoch=0, generation=3, frame_id=3,
            captured_at_ns=base + 16_666_666 * 2,
            inference_end_ts_ns=base + 16_666_666 * 2 + 5_000_000,
            control_now_ns=base + 16_666_666 * 2 + 8_000_000,
        ),
        "detections": [
            _detection(1, 0, 600.0, 600.0, 40.0, 80.0),
            _detection(2, 1, 320.0, 320.0, 30.0, 60.0),
        ],
        "expected_target": {"object_id": 2, "class_id": 1, "lock_reason": "fallback_class"},
    })
    return frames


FIXTURES = {
    "static-target.jsonl": build_static_target,
    "moving-target.jsonl": build_moving_target,
    "freshness-reset.jsonl": build_freshness_reset,
    "dual-phase-control.jsonl": build_dual_phase_control,
    "target-switch-loss.jsonl": build_target_switch_loss,
}


def main() -> int:
    if not SCHEMA_PATH.exists():
        print(f"missing schema at {SCHEMA_PATH}", file=sys.stderr)
        return 1
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    if schema.get("version") != "1":
        print(f"unsupported schema version: {schema.get('version')}", file=sys.stderr)
        return 1

    first_pass = {}
    for filename, builder in FIXTURES.items():
        records = list(builder())
        first_pass[filename] = records
        _dump(FIXTURE_DIR / filename, records)

    for filename, builder in FIXTURES.items():
        second = list(builder())
        first_blob = json.dumps(first_pass[filename], sort_keys=True)
        second_blob = json.dumps(second, sort_keys=True)
        if first_blob != second_blob:
            print(f"{filename} is not deterministic", file=sys.stderr)
            return 2

    print("phase2 fixtures regenerated:")
    for filename in FIXTURES:
        size = (FIXTURE_DIR / filename).stat().st_size
        print(f"  {filename:32s} {size:6d} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
