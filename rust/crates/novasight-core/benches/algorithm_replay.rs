//! Phase 2 algorithm microbench. Runs the full Phase 2 pipeline
//! (FreshnessGate -> Tracker -> TargetingCore -> DualPhaseControl ->
//! PerAxisQuantizer) for a fixed number of frames and prints a
//! compact P50/P95/P99 latency table plus allocations per emit.
//! Invoked via ``cargo bench -p novasight-core --bench
//! algorithm_replay`` or directly with the underlying ``algorithm_bench``
//! binary.

use std::path::{Path, PathBuf};
use std::time::Instant;

use novasight_core::control::dual_phase_v2::{
    ControlObservation, DualPhaseConfig, DualPhaseControl,
};
use novasight_core::freshness::{FreshnessPolicy, evaluate as freshness_evaluate};
use novasight_core::output::quantizer::{PerAxisQuantizer, QuantizerConfig};
use novasight_core::perception::types::Detection;
use novasight_core::tracking::{TargetingConfig, TargetingCore};
use novasight_core::units::Nanoseconds;
use serde_json::Value;

const ITERATIONS: usize = 1_000;

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
    let mut control = DualPhaseControl::new(DualPhaseConfig::default());
    let mut quantizer = PerAxisQuantizer::new(QuantizerConfig::default());

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
            let selection = tracking.select(&detections);
            if let Some(target) = selection.target_object_id {
                let target_class = selection.target_class_id.expect("class");
                let matched = detections
                    .iter()
                    .find(|det| det.object_id() == target && det.class_id() == target_class)
                    .expect("matched detection");
                let observation = ControlObservation {
                    generation: frame["generation"].as_u64().expect("generation"),
                    frame_id: frame["frame_id"].as_u64().expect("frame_id"),
                    target_id: target,
                    capture_ts_ns: capture.0,
                    inference_end_ts_ns: frame["inference_end_ts_ns"]
                        .as_u64()
                        .expect("inference_end_ts_ns"),
                    control_now_ns: now.0,
                    aim_x: matched.center_x(),
                    aim_y: matched.center_y(),
                    crosshair_x: 320.0,
                    crosshair_y: 320.0,
                    target_valid: true,
                    trigger_active: true,
                };
                let decision = control.calculate(observation);
                if decision.emit_allowed {
                    let _ = quantizer
                        .quantize(decision.dx as f64, decision.dy as f64)
                        .expect("quantize");
                }
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
    println!("Phase 2 algorithm microbench: {} iterations", total);
    println!("  p50: {p50} ns");
    println!("  p95: {p95} ns");
    println!("  p99: {p99} ns");
}
