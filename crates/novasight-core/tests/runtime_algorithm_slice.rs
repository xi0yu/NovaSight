//! Phase 2 algorithm-slice end-to-end test. Drives the full pipeline
//! `FreshnessGate -> Tracker -> TargetingCore -> DualPhaseControl` against the
//! captured regression fixtures. The runtime session is *not* the
//! subject of this test; we instantiate each algorithm directly
//! so the Phase 1 contract tests are not disturbed.

use std::path::PathBuf;

use novasight_core::controller::{
    BlockReason, ControlMode, ControlObservation, DualPhaseConfig, DualPhaseControl,
};
use novasight_core::freshness::{FreshnessPolicy, evaluate as freshness_evaluate};
use novasight_core::perception::types::Detection;
use novasight_core::tracking::{TargetingConfig, TargetingCore};
use novasight_core::units::Nanoseconds;
use serde_json::Value;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn load_records(name: &str) -> Vec<Value> {
    let path = fixture_dir().join(name);
    let body = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("missing fixture {name}: {err}"));
    body.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).expect("valid JSON"))
        .collect()
}

fn detection_from_value(value: &Value) -> Detection {
    Detection::new(
        value["object_id"].as_u64().expect("object_id"),
        value["class_id"].as_u64().expect("class_id") as u32,
        value["x"].as_f64().expect("x") as f32,
        value["y"].as_f64().expect("y") as f32,
        value["w"].as_f64().expect("w") as f32,
        value["h"].as_f64().expect("h") as f32,
        value["confidence"].as_f64().expect("confidence") as f32,
    )
    .expect("detection valid")
}

#[test]
fn freshness_records_route_through_freshness_gate() {
    let records = load_records("freshness-reset.jsonl");
    assert!(!records.is_empty());
    let policy = FreshnessPolicy::new(55.0).expect("policy");
    for record in &records {
        let frame = &record["frame"];
        let capture = Nanoseconds(frame["captured_at_ns"].as_u64().expect("captured_at_ns"));
        let now = Nanoseconds(frame["control_now_ns"].as_u64().expect("control_now_ns"));
        let outcome = freshness_evaluate(&policy, capture.0, now.0);
        let expected_admit = record["outcome"]["admit"].as_bool().expect("admit");
        assert_eq!(outcome.admit, expected_admit, "fixture admit flag");
    }
}

#[test]
fn static_target_pipeline_drives_freshness_targeting_and_dual_phase() {
    let records = load_records("static-target.jsonl");
    assert!(!records.is_empty());
    let policy = FreshnessPolicy::new(55.0).expect("policy");
    for record in &records {
        let frame = &record["frame"];
        let capture = Nanoseconds(frame["captured_at_ns"].as_u64().expect("captured_at_ns"));
        let now = Nanoseconds(frame["control_now_ns"].as_u64().expect("control_now_ns"));
        assert!(freshness_evaluate(&policy, capture.0, now.0).admit);
        let detections: Vec<Detection> = record["detections"]
            .as_array()
            .expect("detections array")
            .iter()
            .map(detection_from_value)
            .collect();
        let mut tracking = TargetingCore::new(TargetingConfig::default());
        let selection = tracking.select(&detections, (320.0, 320.0));
        let target = selection.target_object_id.expect("target");
        let target_class = selection.target_class_id.expect("class");
        let matched = detections
            .iter()
            .find(|det| det.object_id() == target && det.class_id() == target_class)
            .expect("matched detection");
        let mut control = DualPhaseControl::new(DualPhaseConfig::default());
        let observation = ControlObservation {
            generation: frame["generation"].as_u64().expect("generation"),
            target_id: target,
            capture_ts_ns: capture.0,
            control_now_ns: now.0,
            aim_x: matched.center_x(),
            aim_y: matched.center_y(),
            crosshair_x: 320.0,
            crosshair_y: 320.0,
            detection_confidence: f64::from(matched.confidence()),
            track_confidence: tracking
                .locked()
                .map_or(0.0, |track| track.identity_confidence),
            target_valid: true,
            trigger_active: true,
        };
        let decision = control.calculate(observation);
        assert_eq!(decision.emit_allowed, decision.dx != 0 || decision.dy != 0);
    }
}

#[test]
fn moving_target_records_emit_typed_decisions() {
    let records = load_records("moving-target.jsonl");
    assert!(!records.is_empty());
    let mut tracking = TargetingCore::new(TargetingConfig::default());
    let mut control = DualPhaseControl::new(DualPhaseConfig::default());
    for record in &records {
        let frame = &record["frame"];
        let detections: Vec<Detection> = record["detections"]
            .as_array()
            .expect("detections array")
            .iter()
            .map(detection_from_value)
            .collect();
        let selection = tracking.select(&detections, (320.0, 320.0));
        if let Some(target) = selection.target_object_id {
            let target_class = selection.target_class_id.expect("class");
            let matched = detections
                .iter()
                .find(|det| det.object_id() == target && det.class_id() == target_class)
                .expect("matched detection");
            let observation = ControlObservation {
                generation: frame["generation"].as_u64().expect("generation"),
                target_id: target,
                capture_ts_ns: frame["captured_at_ns"].as_u64().expect("captured_at_ns"),
                control_now_ns: frame["control_now_ns"].as_u64().expect("control_now_ns"),
                aim_x: matched.center_x(),
                aim_y: matched.center_y(),
                crosshair_x: 320.0,
                crosshair_y: 320.0,
                detection_confidence: f64::from(matched.confidence()),
                track_confidence: tracking
                    .locked()
                    .map_or(0.0, |track| track.identity_confidence),
                target_valid: true,
                trigger_active: true,
            };
            let decision = control.calculate(observation);
            assert!(decision.mode == ControlMode::Far || decision.mode == ControlMode::Near);
        }
    }
}

#[test]
fn dual_phase_first_observation_emits_first_decision() {
    let mut control = DualPhaseControl::new(DualPhaseConfig::default());
    let observation = ControlObservation {
        generation: 1,
        target_id: 1,
        capture_ts_ns: 1_000_000_000,
        control_now_ns: 1_008_000_000,
        aim_x: 400.0,
        aim_y: 360.0,
        crosshair_x: 320.0,
        crosshair_y: 320.0,
        detection_confidence: 1.0,
        track_confidence: 1.0,
        target_valid: true,
        trigger_active: true,
    };
    let decision = control.calculate(observation);
    assert!(decision.emit_allowed);
    assert_eq!(decision.block_reason, BlockReason::None);
    assert_eq!(decision.mode, ControlMode::Far);
}
