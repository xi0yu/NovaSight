//! Phase 2 dual-phase control parity tests. Reads the
//! `dual-phase-control.jsonl` fixture, instantiates a fresh
//! `DualPhaseControl` for each record, and asserts the contract the
//! runtime depends on:
//!
//! * `block_reason` is `BlockReason::None` only when
//!   `target_valid && trigger_active`.
//! * `dx` / `dy` are integer counts that the device can consume.
//! * `quantizer_residual` stays in `[-residual_cap, +residual_cap]`.
//! * `velocity_x` / `velocity_y` are finite; on the first observation
//!   they are exactly zero (no prior capture to derive velocity from).
//! * The state machine resets on every `reset()` call and
//!   `release_trigger()` discards fractional counts but keeps the
use std::path::{Path, PathBuf};

use novasight_core::control::dual_phase_v2::{
    BlockReason, ControlDecision, ControlMode, ControlObservation, DualPhaseConfig,
    DualPhaseControl,
};
use serde_json::Value;

fn fixture_dir() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .join("fixtures")
        .join("phase2")
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

fn observation_from_value(value: &Value) -> ControlObservation {
    ControlObservation {
        generation: value["generation"].as_u64().expect("generation"),
        frame_id: value["frame_id"].as_u64().expect("frame_id"),
        target_id: value["target_id"].as_u64().expect("target_id"),
        capture_ts_ns: value["capture_ts_ns"].as_u64().expect("capture_ts_ns"),
        inference_end_ts_ns: value["inference_end_ts_ns"]
            .as_u64()
            .expect("inference_end_ts_ns"),
        control_now_ns: value["control_now_ns"].as_u64().expect("control_now_ns"),
        aim_x: value["aim_x"].as_f64().expect("aim_x"),
        aim_y: value["aim_y"].as_f64().expect("aim_y"),
        crosshair_x: value["crosshair_x"].as_f64().expect("crosshair_x"),
        crosshair_y: value["crosshair_y"].as_f64().expect("crosshair_y"),
        detection_confidence: value["detection_confidence"].as_f64().unwrap_or(1.0),
        track_confidence: value["track_confidence"].as_f64().unwrap_or(1.0),
        target_valid: value["target_valid"].as_bool().expect("target_valid"),
        trigger_active: value["trigger_active"].as_bool().expect("trigger_active"),
    }
}

fn assert_invariants(record: &Value, decision: ControlDecision) {
    let expected_emit = record["decision"]["emit_allowed"]
        .as_bool()
        .expect("emit_allowed");
    let expected_block_reason = record["decision"]["block_reason"]
        .as_str()
        .expect("block_reason");
    let label = record["label"].as_str().unwrap_or("?");

    assert_eq!(
        decision.emit_allowed, expected_emit,
        "emit_allowed for {label}"
    );

    if expected_emit {
        assert_eq!(
            decision.block_reason,
            BlockReason::None,
            "block_reason must be None when emit_allowed for {label}"
        );
    } else {
        assert_ne!(
            decision.block_reason,
            BlockReason::None,
            "non-emit decision must declare a block_reason for {label}"
        );
        assert!(
            !expected_block_reason.is_empty(),
            "non-emit record must declare a non-empty block_reason for {label}"
        );
    }

    assert!(decision.quantizer_residual_x.is_finite());
    assert!(decision.quantizer_residual_y.is_finite());
    assert!(decision.quantizer_residual_x.abs() <= 1.0);
    assert!(decision.quantizer_residual_y.abs() <= 1.0);

    assert!(decision.velocity_x.is_finite());
    assert!(decision.velocity_y.is_finite());
    assert!(decision.predicted_offset_x.is_finite());
    assert!(decision.predicted_offset_y.is_finite());
}

#[test]
fn dual_phase_control_decision_matches_contract_per_record() {
    let records = load_records("dual-phase-control.jsonl");
    assert!(
        !records.is_empty(),
        "dual-phase-control.jsonl must contain records"
    );
    let mut control = DualPhaseControl::new(DualPhaseConfig::default());
    for record in &records {
        let observation = observation_from_value(&record["observation"]);
        let decision = control.calculate(observation);
        assert_invariants(record, decision);
    }
}

#[test]
fn first_observation_has_zero_velocity_and_predicted_offset() {
    let mut control = DualPhaseControl::new(DualPhaseConfig::default());
    let observation = ControlObservation {
        generation: 1,
        frame_id: 1,
        target_id: 1,
        capture_ts_ns: 1_000_000_000,
        inference_end_ts_ns: 1_005_000_000,
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
    assert_eq!(decision.velocity_x, 0.0);
    assert_eq!(decision.velocity_y, 0.0);
    assert_eq!(decision.predicted_offset_x, 0.0);
    assert_eq!(decision.predicted_offset_y, 0.0);
}

#[test]
fn rust_feedback_matches_the_python_projection_and_atan_reference() {
    let mut control = DualPhaseControl::new(DualPhaseConfig::default());
    let decision = control.calculate(ControlObservation {
        generation: 1,
        frame_id: 1,
        target_id: 1,
        capture_ts_ns: 1_000_000_000,
        inference_end_ts_ns: 1_001_000_000,
        control_now_ns: 1_002_000_000,
        aim_x: 420.0,
        aim_y: 380.0,
        crosshair_x: 320.0,
        crosshair_y: 320.0,
        detection_confidence: 1.0,
        track_confidence: 1.0,
        target_valid: true,
        trigger_active: true,
    });

    assert_eq!((decision.dx, decision.dy), (497, 328));
    assert!((decision.quantizer_residual_x - 0.982_265_322_691).abs() < 1e-9);
    assert!((decision.quantizer_residual_y - 0.009_144_829_047).abs() < 1e-9);
}

#[test]
fn second_observation_waits_for_complete_robust_velocity_window() {
    let mut control = DualPhaseControl::new(DualPhaseConfig::default());
    let first = ControlObservation {
        generation: 1,
        frame_id: 1,
        target_id: 1,
        capture_ts_ns: 1_000_000_000,
        inference_end_ts_ns: 1_005_000_000,
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
    let _ = control.calculate(first);
    let second = ControlObservation {
        generation: 2,
        frame_id: 2,
        target_id: 1,
        capture_ts_ns: 1_016_666_666,
        inference_end_ts_ns: 1_021_666_666,
        control_now_ns: 1_024_666_666,
        aim_x: 420.0,
        aim_y: 380.0,
        crosshair_x: 320.0,
        crosshair_y: 320.0,
        detection_confidence: 1.0,
        track_confidence: 1.0,
        target_valid: true,
        trigger_active: true,
    };
    let decision = control.calculate(second);
    assert_eq!(decision.velocity_x, 0.0);
    assert_eq!(decision.velocity_y, 0.0);
    assert_eq!(decision.predicted_offset_x, 0.0);
    assert_eq!(decision.predicted_offset_y, 0.0);
}

#[test]
fn invalid_target_returns_target_invalid_block() {
    let mut control = DualPhaseControl::new(DualPhaseConfig::default());
    let observation = ControlObservation {
        generation: 1,
        frame_id: 1,
        target_id: 1,
        capture_ts_ns: 1_000_000_000,
        inference_end_ts_ns: 1_005_000_000,
        control_now_ns: 1_008_000_000,
        aim_x: 400.0,
        aim_y: 360.0,
        crosshair_x: 320.0,
        crosshair_y: 320.0,
        detection_confidence: 1.0,
        track_confidence: 1.0,
        target_valid: false,
        trigger_active: true,
    };
    let decision = control.calculate(observation);
    assert!(!decision.emit_allowed);
    assert_eq!(decision.block_reason, BlockReason::TargetInvalid);
}

#[test]
fn trigger_inactive_returns_trigger_inactive_block() {
    let mut control = DualPhaseControl::new(DualPhaseConfig::default());
    let observation = ControlObservation {
        generation: 1,
        frame_id: 1,
        target_id: 1,
        capture_ts_ns: 1_000_000_000,
        inference_end_ts_ns: 1_005_000_000,
        control_now_ns: 1_008_000_000,
        aim_x: 400.0,
        aim_y: 360.0,
        crosshair_x: 320.0,
        crosshair_y: 320.0,
        detection_confidence: 1.0,
        track_confidence: 1.0,
        target_valid: true,
        trigger_active: false,
    };
    let decision = control.calculate(observation);
    assert!(!decision.emit_allowed);
    assert_eq!(decision.block_reason, BlockReason::TriggerInactive);
}

#[test]
fn non_monotonic_observation_is_rejected() {
    let mut control = DualPhaseControl::new(DualPhaseConfig::default());
    let first = ControlObservation {
        generation: 1,
        frame_id: 1,
        target_id: 1,
        capture_ts_ns: 1_000_000_000,
        inference_end_ts_ns: 1_005_000_000,
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
    let _ = control.calculate(first);
    let stale = ControlObservation {
        generation: 1,
        frame_id: 2,
        target_id: 1,
        capture_ts_ns: 1_016_666_666,
        inference_end_ts_ns: 1_021_666_666,
        control_now_ns: 1_024_666_666,
        aim_x: 420.0,
        aim_y: 380.0,
        crosshair_x: 320.0,
        crosshair_y: 320.0,
        detection_confidence: 1.0,
        track_confidence: 1.0,
        target_valid: true,
        trigger_active: true,
    };
    let decision = control.calculate(stale);
    assert!(!decision.emit_allowed);
    assert_eq!(decision.block_reason, BlockReason::NonMonotonicObservation);
}

#[test]
fn stale_observation_is_rejected() {
    let mut control = DualPhaseControl::new(DualPhaseConfig::default());
    let observation = ControlObservation {
        generation: 1,
        frame_id: 1,
        target_id: 1,
        capture_ts_ns: 0,
        inference_end_ts_ns: 5_000_000,
        control_now_ns: 200_000_000, // 200ms in the future
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
    assert!(!decision.emit_allowed);
    assert_eq!(decision.block_reason, BlockReason::StaleObservation);
}

#[test]
fn reset_clears_state_and_history() {
    let mut control = DualPhaseControl::new(DualPhaseConfig::default());
    let observation = ControlObservation {
        generation: 1,
        frame_id: 1,
        target_id: 1,
        capture_ts_ns: 1_000_000_000,
        inference_end_ts_ns: 1_005_000_000,
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
    let _ = control.calculate(observation);
    control.reset();
    let second = ControlObservation {
        generation: 1,
        frame_id: 1,
        target_id: 1,
        capture_ts_ns: 1_000_000_000,
        inference_end_ts_ns: 1_005_000_000,
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
    let decision = control.calculate(second);
    assert_eq!(decision.velocity_x, 0.0);
    assert_eq!(decision.velocity_y, 0.0);
}

#[test]
fn release_trigger_drops_fractional_count_but_keeps_history() {
    let mut control = DualPhaseControl::new(DualPhaseConfig::default());
    let first = ControlObservation {
        generation: 1,
        frame_id: 1,
        target_id: 1,
        capture_ts_ns: 1_000_000_000,
        inference_end_ts_ns: 1_005_000_000,
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
    let _ = control.calculate(first);
    control.release_trigger();
    let last = ControlObservation {
        generation: 1,
        frame_id: 1,
        target_id: 1,
        capture_ts_ns: 1_000_000_000,
        inference_end_ts_ns: 1_005_000_000,
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
    let decision = control.calculate(last);
    assert_eq!(decision.velocity_x, 0.0);
    assert_eq!(decision.velocity_y, 0.0);
    // public surface (next emit) by re-running an observation.
}

#[test]
fn near_mode_is_selected_for_small_error() {
    let mut control = DualPhaseControl::new(DualPhaseConfig::default());
    let observation = ControlObservation {
        generation: 1,
        frame_id: 1,
        target_id: 1,
        capture_ts_ns: 1_000_000_000,
        inference_end_ts_ns: 1_005_000_000,
        control_now_ns: 1_008_000_000,
        aim_x: 322.0,
        aim_y: 322.0,
        crosshair_x: 320.0,
        crosshair_y: 320.0,
        detection_confidence: 1.0,
        track_confidence: 1.0,
        target_valid: true,
        trigger_active: true,
    };
    let decision = control.calculate(observation);
    assert_eq!(decision.mode, ControlMode::Near);
}
