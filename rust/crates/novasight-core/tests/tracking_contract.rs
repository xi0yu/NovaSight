//! Phase 2 tracker and targeting contract tests. Pins the deterministic
//! selection and reset edges from `target-switch-loss.jsonl` and a few
//! negative cases. The fixture reference values come from the Python
//! `RuntimeTargetSelector`; only the small subset of edges the Phase 2
//! slice must respect is asserted here.
use std::path::{Path, PathBuf};

use novasight_core::perception::types::Detection;
use novasight_core::tracking::{LockReason, TargetingConfig, TargetingCore};
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

fn detections_from_value(value: &Value) -> Vec<Detection> {
    value
        .as_array()
        .expect("detections array")
        .iter()
        .map(|det| {
            Detection::new(
                det["object_id"].as_u64().expect("object_id"),
                det["class_id"].as_u64().expect("class_id") as u32,
                det["x"].as_f64().expect("x") as f32,
                det["y"].as_f64().expect("y") as f32,
                det["w"].as_f64().expect("w") as f32,
                det["h"].as_f64().expect("h") as f32,
                det["confidence"].as_f64().expect("confidence") as f32,
            )
            .expect("detection valid")
        })
        .collect()
}

fn reason_from_str(value: &str) -> LockReason {
    match value {
        "preferred_class" => LockReason::PreferredClass,
        "fallback_class" => LockReason::FallbackClass,

        other => panic!("unknown lock_reason {other}"),
    }
}

#[test]
fn target_switch_loss_fixture_matches_targeting_core() {
    let records = load_records("target-switch-loss.jsonl");
    assert!(
        records.len() >= 3,
        "fixture must cover lock, challenger_close, challenger_commit"
    );

    let mut core = TargetingCore::new(TargetingConfig::default());

    for record in &records {
        let detections = detections_from_value(&record["detections"]);
        let selection = core.select(&detections);
        let expected = &record["expected_target"];
        let expected_object_id = expected["object_id"].as_u64().expect("object_id");
        let expected_class_id = expected["class_id"].as_u64().expect("class_id") as u32;
        let expected_reason =
            reason_from_str(expected["lock_reason"].as_str().expect("lock_reason"));
        assert_eq!(
            selection.target_object_id,
            Some(expected_object_id),
            "object_id for {label}",
            label = record["label"]
        );
        assert_eq!(
            selection.target_class_id,
            Some(expected_class_id),
            "class_id for {label}",
            label = record["label"]
        );
        assert_eq!(
            selection.lock_reason,
            Some(expected_reason),
            "lock_reason for {label}",
            label = record["label"]
        );
    }
}

#[test]
fn empty_detections_clear_the_lock_and_increment_lost_count() {
    let mut core = TargetingCore::new(TargetingConfig::default());
    let head = Detection::new(1, 0, 300.0, 280.0, 40.0, 80.0, 0.9).expect("head");
    let first = core.select(&[head]);
    assert!(first.target_object_id.is_some());
    let empty = core.select(&[]);
    assert!(empty.target_object_id.is_none());
    assert_eq!(empty.lost_count, 1);
    assert!(core.locked().is_none());
}

#[test]
fn reset_clears_state_and_history() {
    let mut core = TargetingCore::new(TargetingConfig::default());
    let head = Detection::new(1, 0, 300.0, 280.0, 40.0, 80.0, 0.9).expect("head");
    core.select(&[head]);
    assert!(core.history_iter().next().is_some());
    core.reset();
    assert!(core.history_iter().next().is_none());
    assert!(core.locked().is_none());
    assert_eq!(core.lost_count(), 0);
}

#[test]
fn below_confidence_detections_are_filtered_before_targeting() {
    let mut core = TargetingCore::new(TargetingConfig {
        min_confidence: 0.5,
        ..TargetingConfig::default()
    });
    let weak = Detection::new(1, 0, 300.0, 280.0, 40.0, 80.0, 0.1).expect("weak");
    let selection = core.select(&[weak]);
    assert!(selection.target_object_id.is_none());
    assert_eq!(selection.candidates, 1);
    assert_eq!(selection.inside_fov, 0);
}

#[test]
fn head_movement_under_debounce_keeps_lock_with_held_by_debounce_reason() {
    let mut core = TargetingCore::new(TargetingConfig {
        debounce_distance_px: 64.0,
        ..TargetingConfig::default()
    });
    let frame1 = vec![Detection::new(1, 0, 300.0, 300.0, 40.0, 80.0, 0.9).expect("d1")];
    let first = core.select(&frame1);
    assert_eq!(first.lock_reason, Some(LockReason::PreferredClass));
    // Head moved 20 px, still within debounce window.
    let frame2 = vec![Detection::new(1, 0, 320.0, 300.0, 40.0, 80.0, 0.9).expect("d2")];
    let second = core.select(&frame2);
    assert_eq!(second.target_object_id, Some(1));
    assert_eq!(second.lock_reason, Some(LockReason::PreferredClass));
}

#[test]
fn history_is_bounded() {
    let mut core = TargetingCore::new(TargetingConfig::default());
    for index in 0..(novasight_core::tracking::DEFAULT_HISTORY_LIMIT * 4) {
        let x = 300.0 + index as f32;
        let detection = Detection::new(1, 0, x, 300.0, 40.0, 80.0, 0.9).expect("d");
        core.select(&[detection]);
    }
    let count = core.history_iter().count();
    assert!(count <= novasight_core::tracking::DEFAULT_HISTORY_LIMIT);
}

#[test]
fn lost_track_cannot_produce_a_target_object_id() {
    let mut core = TargetingCore::new(TargetingConfig::default());
    let head = Detection::new(1, 0, 300.0, 280.0, 40.0, 80.0, 0.9).expect("head");
    core.select(&[head]);
    // Two empty frames in a row force the lock to Lost; the next
    // admissible frame restarts the targeting state cleanly.
    core.select(&[]);
    core.select(&[]);
    let body = Detection::new(2, 1, 320.0, 360.0, 30.0, 60.0, 0.9).expect("body");
    let next = core.select(&[body]);
    assert_eq!(next.target_object_id, Some(2));
    assert_eq!(next.target_class_id, Some(1));
}
