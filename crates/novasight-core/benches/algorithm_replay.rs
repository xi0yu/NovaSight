//! Runtime algorithm microbench. Runs the full runtime algorithm path
//! (FreshnessGate -> Tracker -> TargetingCore -> ContinuousControl) for a fixed
//! number of frames and prints a
//! compact P50/P95/P99 latency table plus allocations per emit.
//! Invoked via ``cargo bench -p novasight-core --bench
//! algorithm_replay`` or directly with the underlying ``algorithm_bench``
//! binary.

use std::path::PathBuf;
use std::time::Instant;

use novasight_core::controller::{ContinuousControl, ContinuousControlConfig, ControlObservation};
use novasight_core::freshness::{FreshnessPolicy, evaluate as freshness_evaluate};
use novasight_core::perception::types::Detection;
use novasight_core::tracking::{TargetingConfig, TargetingCore};
use novasight_core::units::Nanoseconds;
use serde_json::Value;

const ITERATIONS: usize = 1_000;

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

fn percentile(samples: &mut [u128], q: f64) -> u128 {
    samples.sort_unstable();
    let idx = ((samples.len() as f64 - 1.0) * q) as usize;
    samples[idx]
}

fn main() {
    let records = load_records("static-target.jsonl");
    let policy = FreshnessPolicy::new(55.0).expect("policy");
    let mut tracking = TargetingCore::new(TargetingConfig::default());
    let mut control = ContinuousControl::new(ContinuousControlConfig::default());

    let mut samples: Vec<u128> = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        for record in &records {
            let frame = &record["frame"];
            let capture = Nanoseconds(frame["captured_at_ns"].as_u64().expect("captured_at_ns"));
            let now = Nanoseconds(frame["control_now_ns"].as_u64().expect("control_now_ns"));
            let detections: Vec<Detection> = record["detections"]
                .as_array()
                .expect("detections array")
                .iter()
                .map(detection_from_value)
                .collect();
            let start = Instant::now();
            let _ = freshness_evaluate(&policy, capture.0, now.0);
            let selection = tracking.select_at(&detections, (320.0, 320.0), capture.0);
            if let Some(target) = selection.target_object_id {
                let target_class = selection.target_class_id.expect("class");
                assert!(
                    detections
                        .iter()
                        .any(|det| det.object_id() == target && det.class_id() == target_class),
                    "selected target must belong to the frame"
                );
                let observation = ControlObservation {
                    generation: frame["generation"].as_u64().expect("generation"),
                    target_id: target,
                    capture_ts_ns: capture.0,
                    control_now_ns: now.0,
                    aim_x: selection.target_aim_x.expect("aim x"),
                    aim_y: selection.target_aim_y.expect("aim y"),
                    crosshair_x: 320.0,
                    crosshair_y: 320.0,
                    detection_confidence: 1.0,
                    track_confidence: 1.0,
                    target_valid: true,
                    trigger_active: true,
                };
                let _decision = control.calculate(observation);
            }
            samples.push(start.elapsed().as_nanos());
        }
    }
    let total = samples.len();
    let p50 = percentile(&mut samples.clone(), 0.50);
    let mut sorted = samples.clone();
    sorted.sort_unstable();
    let p95 = percentile(&mut sorted.clone(), 0.95);
    let p99 = percentile(&mut sorted, 0.99);
    println!("Runtime algorithm microbench: {} iterations", total);
    println!("  p50: {p50} ns");
    println!("  p95: {p95} ns");
    println!("  p99: {p99} ns");

    let mut crowded_tracking = TargetingCore::new(TargetingConfig::default());
    let mut crowded_samples = Vec::with_capacity(ITERATIONS);
    for iteration in 0..ITERATIONS {
        let detections: Vec<Detection> = (0..16)
            .map(|index| {
                let column = index % 4;
                let row = index / 4;
                Detection::new(
                    (iteration * 16 + index) as u64,
                    0,
                    260.0 + column as f32 * 40.0 + (iteration % 2) as f32,
                    240.0 + row as f32 * 40.0,
                    30.0,
                    60.0,
                    0.9,
                )
                .expect("crowded detection")
            })
            .collect();
        let start = Instant::now();
        let selection = crowded_tracking.select_at(
            &detections,
            (320.0, 320.0),
            10_000_000_000 + iteration as u64 * 1_000_000,
        );
        if iteration > 0 {
            assert!(selection.target_track_id.is_some());
        }
        crowded_samples.push(start.elapsed().as_nanos());
    }
    let crowded_p50 = percentile(&mut crowded_samples.clone(), 0.50);
    let mut crowded_sorted = crowded_samples;
    crowded_sorted.sort_unstable();
    let crowded_p95 = percentile(&mut crowded_sorted.clone(), 0.95);
    let crowded_p99 = percentile(&mut crowded_sorted, 0.99);
    println!("16x16 association microbench: {ITERATIONS} iterations");
    println!("  p50: {crowded_p50} ns");
    println!("  p95: {crowded_p95} ns");
    println!("  p99: {crowded_p99} ns");
}
