//! Algorithm-slice end-to-end test. Drives the full pipeline
//! `FreshnessGate -> Tracker -> TargetingCore -> ContinuousControl` against the
//! captured regression fixtures. The runtime session is *not* the
//! subject of this test; we instantiate each algorithm directly
//! so the Phase 1 contract tests are not disturbed.

use std::path::PathBuf;

use novasight_core::controller::{
    BlockReason, ContinuousControl, ContinuousControlConfig, ControlMode, ControlObservation,
};
use novasight_core::freshness::{FreshnessPolicy, evaluate as freshness_evaluate};
use novasight_core::perception::types::Detection;
use novasight_core::tracking::{TargetingConfig, TargetingCore};
use novasight_core::units::Nanoseconds;
use novasight_core::{
    AlgorithmScoreConfig, AlgorithmTraceSample, CountResponseModel, PredictionTruthConfig,
    PredictionTruthProjection, PredictionTruthSample, estimate_count_response_lag,
    score_algorithm_trace, score_prediction_truth,
};
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
fn static_target_pipeline_drives_freshness_targeting_and_control() {
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
        let mut control = ContinuousControl::new(ContinuousControlConfig::default());
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
    let mut control = ContinuousControl::new(ContinuousControlConfig::default());
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
            assert_eq!(decision.mode, ControlMode::Continuous);
        }
    }
}

#[test]
fn control_first_observation_emits_first_decision() {
    let mut control = ContinuousControl::new(ContinuousControlConfig::default());
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
    assert_eq!(decision.mode, ControlMode::Continuous);
}

#[test]
fn closed_loop_algorithm_score_tracks_visual_convergence() {
    let control_config = ContinuousControlConfig::default();
    let focal_x = (control_config.source_width as f64 * 0.5)
        / (control_config.projection_fov_x_deg.to_radians() * 0.5).tan();
    let observation_px_per_count =
        focal_x * (std::f64::consts::TAU / control_config.projection_counts_per_360).tan();
    let mut targeting = TargetingCore::new(TargetingConfig::default());
    let mut control = ContinuousControl::new(control_config);
    let mut true_error_x = 100.0;
    let mut delayed_errors = [true_error_x; 3];
    let mut trace = Vec::new();
    let mut prediction_truth_trace = Vec::new();

    for generation in 1..=80_u64 {
        let observed_error_x = delayed_errors[0];
        delayed_errors.rotate_left(1);
        let capture_ts_ns = 1_000_000_000 + generation * 8_333_333;
        let box_height = 40.0;
        let detection = Detection::new(
            generation,
            0,
            (320.0 + observed_error_x - 20.0) as f32,
            (320.0 - box_height * 0.22) as f32,
            40.0,
            box_height as f32,
            0.95,
        )
        .expect("valid detection");
        let selection = targeting.select_at(&[detection], (320.0, 320.0), capture_ts_ns);
        let decision = control.calculate(ControlObservation {
            generation,
            target_id: selection.target_track_id.map_or(0, |track_id| track_id.0),
            capture_ts_ns,
            control_now_ns: capture_ts_ns + 4_000_000,
            aim_x: selection.target_aim_x.unwrap_or(320.0),
            aim_y: selection.target_aim_y.unwrap_or(320.0),
            crosshair_x: 320.0,
            crosshair_y: 320.0,
            detection_confidence: selection.target_detection_confidence.map_or(0.0, f64::from),
            track_confidence: selection.target_identity_confidence.unwrap_or(0.0),
            target_valid: selection.target_track_id.is_some(),
            trigger_active: true,
        });
        true_error_x -= f64::from(decision.dx) * observation_px_per_count;
        delayed_errors[2] = true_error_x;
        trace.push(AlgorithmTraceSample::from_control_decision(&decision));
        prediction_truth_trace.push(PredictionTruthSample::from_control_decision(&decision));
    }

    let score = score_algorithm_trace(&trace, AlgorithmScoreConfig::default());
    let prediction_truth = score_prediction_truth(
        &prediction_truth_trace,
        PredictionTruthConfig {
            horizons_ms: vec![8.333333, 16.666666],
            projection: PredictionTruthProjection::Capped,
            ..PredictionTruthConfig::default()
        },
    );

    assert_eq!(score.total_samples, 80);
    assert_eq!(prediction_truth.total_samples, 80);
    assert!(
        prediction_truth
            .horizons
            .iter()
            .all(|score| score.sample_pairs > 0 && score.mae_px.is_finite()),
        "prediction truth report should score real control decisions; report={prediction_truth:?}"
    );
    assert_eq!(score.target_switches, 0);
    assert!(score.emitted_commands > 0);
    assert!(
        score.final_error_px < score.initial_error_px * 0.05,
        "final error should be materially lower; score={score:?}"
    );
    assert!(
        score.mean_tail_error_px < 1.0,
        "tail error should stay near the aim point; score={score:?}"
    );
    assert!(
        score.max_overshoot_px < 2.0,
        "default control should not hide a large overshoot; score={score:?}"
    );
    assert!(
        score.settle_generation.is_some(),
        "trace should reach the configured settle window; score={score:?}"
    );

    let response_lag = estimate_count_response_lag(
        &trace,
        CountResponseModel {
            px_per_count_x: observation_px_per_count,
            px_per_count_y: observation_px_per_count,
            min_lag_samples: 1,
            max_lag_samples: 5,
        },
    )
    .expect("count response lag estimate");

    assert_eq!(
        response_lag.best.lag_samples, 3,
        "synthetic visual feedback delay should be recovered; estimate={response_lag:?}"
    );
    assert!(
        response_lag.best.rms_residual_px < 1e-4,
        "stationary synthetic response should leave no unexplained residual; estimate={response_lag:?}"
    );
}
