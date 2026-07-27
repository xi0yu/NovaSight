//! Phase 2 tracker and targeting contract tests. Pins the deterministic
//! selection and reset edges from `target-switch-loss.jsonl` and a few
//! negative cases. The fixtures preserve the target-selection edges the
//! runtime must continue to respect.
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use novasight_core::perception::types::Detection;
use novasight_core::tracking::{KalmanConfig, LockReason, TargetingConfig, TargetingCore};
use serde_json::Value;

const OBSERVATION_CENTER: (f64, f64) = (320.0, 320.0);

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

    let first_detections = detections_from_value(&records[0]["detections"]);
    let first_captured_at_ns = records[0]["frame"]["captured_at_ns"]
        .as_u64()
        .expect("captured_at_ns");
    assert!(
        core.select_at(&first_detections, OBSERVATION_CENTER, first_captured_at_ns)
            .target_object_id
            .is_none(),
        "multiple fresh candidates require a confirming observation"
    );

    for record in &records {
        let detections = detections_from_value(&record["detections"]);
        let captured_at_ns = record["frame"]["captured_at_ns"]
            .as_u64()
            .expect("captured_at_ns");
        let selection = core.select_at(&detections, OBSERVATION_CENTER, captured_at_ns);
        let expected = &record["expected_target"];
        let expected_object_id = expected["object_id"].as_u64();
        let expected_class_id = expected["class_id"].as_u64().map(|value| value as u32);
        let expected_reason = expected["lock_reason"].as_str().map(reason_from_str);
        assert_eq!(
            selection.target_object_id,
            expected_object_id,
            "object_id for {label}",
            label = record["label"]
        );
        assert_eq!(
            selection.target_class_id,
            expected_class_id,
            "class_id for {label}",
            label = record["label"]
        );
        assert_eq!(
            selection.lock_reason,
            expected_reason,
            "lock_reason for {label}",
            label = record["label"]
        );
    }
}

#[test]
fn empty_detections_hold_lock_identity_without_emitting_a_stale_target() {
    let mut core = TargetingCore::new(TargetingConfig::default());
    let head = Detection::new(1, 0, 300.0, 280.0, 40.0, 80.0, 0.9).expect("head");
    let first = core.select(&[head], OBSERVATION_CENTER);
    assert!(first.target_object_id.is_some());
    let empty = core.select(&[], OBSERVATION_CENTER);
    assert!(empty.target_object_id.is_none());
    assert_eq!(empty.lost_count, 1);
    assert_eq!(core.locked().expect("grace lock").missed_frames, 1);
}

#[test]
fn detection_reacquired_inside_grace_keeps_the_same_track_id() {
    let mut core = TargetingCore::new(TargetingConfig {
        track_max_age: 2,
        ..TargetingConfig::default()
    });
    let first = Detection::new(1, 0, 280.0, 280.0, 80.0, 100.0, 0.9).expect("first");
    let track_id = core
        .select(&[first], OBSERVATION_CENTER)
        .target_track_id
        .expect("track id");
    assert!(
        core.select(&[], OBSERVATION_CENTER)
            .target_track_id
            .is_none()
    );
    let reacquired = Detection::new(2, 0, 284.0, 280.0, 80.0, 100.0, 0.9).expect("reacquired");
    let selection = core.select(&[reacquired], OBSERVATION_CENTER);
    assert_eq!(selection.target_track_id, Some(track_id));
    assert_eq!(selection.lost_count, 0);
}

#[test]
fn smooth_visual_motion_keeps_identity_and_does_not_rebuild_control_state() {
    let mut core = TargetingCore::new(TargetingConfig::default());
    let mut stable_track_id = None;

    for (frame, x) in [240.0_f32, 256.0, 272.0, 288.0, 304.0]
        .into_iter()
        .enumerate()
    {
        let detection =
            Detection::new(frame as u64, 0, x, 240.0, 80.0, 160.0, 0.9).expect("moving target");
        let selected = core.select_at(
            &[detection],
            OBSERVATION_CENTER,
            1_000_000_000 + frame as u64 * 8_333_333,
        );
        let track_id = selected.target_track_id.expect("selected track");
        if let Some(expected) = stable_track_id {
            assert_eq!(track_id, expected, "frame {frame} rebuilt target identity");
            assert!(
                !selected.target_rebuilt,
                "frame {frame} unnecessarily resets controller history"
            );
        } else {
            stable_track_id = Some(track_id);
        }
    }
}

#[test]
fn sticky_lock_is_not_disabled_by_an_unrelated_pixel_debounce_threshold() {
    let mut core = TargetingCore::new(TargetingConfig {
        class_priority: Vec::new(),
        selection_class_weight: 0.0,
        selection_distance_weight: 1.0,
        sticky_bias: 0.90,
        switch_min_preference_advantage: 0.0,
        switch_min_continuity_score: 0.0,
        switch_delay_ms: 0.0,
        kalman: KalmanConfig {
            nis_threshold: 1_000_000.0,
            nis_hard_reject: 1_000_000.0,
            min_prediction_confidence: 0.0,
            ..KalmanConfig::default()
        },
        ..TargetingConfig::default()
    });
    let initial = [
        Detection::new(10, 0, 280.0, 280.0, 80.0, 100.0, 0.9).expect("locked"),
        Detection::new(20, 1, 360.0, 280.0, 80.0, 100.0, 0.9).expect("challenger"),
    ];
    assert!(
        core.select_at(&initial, OBSERVATION_CENTER, 1_000_000_000)
            .target_track_id
            .is_none()
    );
    let locked = core.select_at(&initial, OBSERVATION_CENTER, 1_010_000_000);
    let locked_track = locked.target_track_id.expect("confirmed lock");

    let moved = [
        Detection::new(11, 0, 350.0, 280.0, 80.0, 100.0, 0.9).expect("locked moved"),
        Detection::new(21, 1, 300.0, 280.0, 80.0, 100.0, 0.9).expect("challenger close"),
    ];
    let selected = core.select_at(&moved, OBSERVATION_CENTER, 1_020_000_000);

    assert_eq!(selected.target_track_id, Some(locked_track));
    assert_eq!(selected.target_class_id, Some(0));
}

#[test]
fn equal_scores_tie_break_by_stable_track_id_not_frame_local_object_id() {
    let mut core = TargetingCore::new(TargetingConfig {
        class_priority: Vec::new(),
        selection_class_weight: 0.0,
        selection_distance_weight: 1.0,
        sticky_bias: 0.0,
        ..TargetingConfig::default()
    });
    let first = [
        Detection::new(10, 0, 270.0, 280.0, 80.0, 100.0, 0.9).expect("left"),
        Detection::new(20, 1, 290.0, 280.0, 80.0, 100.0, 0.9).expect("right"),
    ];
    core.select_at(&first, OBSERVATION_CENTER, 1_000_000_000);
    let confirmed = core.select_at(&first, OBSERVATION_CENTER, 1_010_000_000);
    let stable_track = confirmed.target_track_id.expect("stable tie winner");

    let reordered_ids = [
        Detection::new(99, 0, 270.0, 280.0, 80.0, 100.0, 0.9).expect("left"),
        Detection::new(1, 1, 290.0, 280.0, 80.0, 100.0, 0.9).expect("right"),
    ];
    let selected = core.select_at(&reordered_ids, OBSERVATION_CENTER, 1_020_000_000);

    assert_eq!(selected.target_track_id, Some(stable_track));
    assert_eq!(selected.target_class_id, Some(0));
}

#[test]
fn reset_clears_state_and_history() {
    let mut core = TargetingCore::new(TargetingConfig::default());
    let head = Detection::new(1, 0, 300.0, 280.0, 40.0, 80.0, 0.9).expect("head");
    core.select(&[head], OBSERVATION_CENTER);
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
    let selection = core.select(&[weak], OBSERVATION_CENTER);
    assert!(selection.target_object_id.is_none());
    assert_eq!(selection.candidates, 1);
    assert_eq!(selection.inside_fov, 0);
}

#[test]
fn explicit_class_filter_is_independent_from_class_priority() {
    let mut core = TargetingCore::new(TargetingConfig {
        class_priority: vec![0, 1],
        allowed_class_ids: Some(BTreeSet::from([2])),
        ..TargetingConfig::default()
    });
    let preferred_but_blocked =
        Detection::new(1, 0, 300.0, 280.0, 40.0, 80.0, 0.9).expect("blocked");
    let unranked_but_allowed =
        Detection::new(2, 2, 300.0, 280.0, 40.0, 80.0, 0.9).expect("allowed");

    let selection = core.select(
        &[preferred_but_blocked, unranked_but_allowed],
        OBSERVATION_CENTER,
    );

    assert_eq!(selection.target_object_id, Some(2));
    assert_eq!(selection.inside_fov, 1);
    assert_eq!(selection.rejected_by_class, 1);
    assert_eq!(selection.rejected_class_ids, vec![0]);
}

#[test]
fn empty_class_allowlist_disables_target_selection() {
    let mut core = TargetingCore::new(TargetingConfig {
        allowed_class_ids: Some(BTreeSet::new()),
        ..TargetingConfig::default()
    });
    let target = Detection::new(1, 0, 300.0, 280.0, 40.0, 80.0, 0.9).expect("target");

    let selection = core.select(&[target], OBSERVATION_CENTER);

    assert!(selection.target_object_id.is_none());
    assert_eq!(selection.inside_fov, 0);
}

#[test]
fn first_lock_is_ranked_from_declared_observation_center_not_origin() {
    let mut core = TargetingCore::new(TargetingConfig {
        target_fov_radius_px: 1_000.0,
        ..TargetingConfig::default()
    });
    let near_origin = Detection::new(1, 0, 90.0, 90.0, 20.0, 20.0, 0.9).expect("origin");
    let near_center = Detection::new(2, 0, 290.0, 290.0, 20.0, 20.0, 0.9).expect("center");

    let selected = core.select(
        &[near_origin.clone(), near_center.clone()],
        OBSERVATION_CENTER,
    );
    assert!(selected.target_object_id.is_none());
    let selected = core.select(&[near_origin, near_center], OBSERVATION_CENTER);

    assert_eq!(selected.target_object_id, Some(2));
}

#[test]
fn target_fov_radius_is_a_real_radial_admission_gate() {
    let mut core = TargetingCore::new(TargetingConfig {
        target_fov_radius_px: 180.0,
        ..TargetingConfig::default()
    });
    let outside = Detection::new(1, 0, 530.0, 310.0, 20.0, 20.0, 0.99).expect("outside");
    let inside = Detection::new(2, 0, 310.0, 310.0, 20.0, 20.0, 0.60).expect("inside");

    let selected = core.select(&[outside.clone(), inside.clone()], OBSERVATION_CENTER);
    assert!(selected.target_object_id.is_none());
    let selected = core.select(&[outside, inside], OBSERVATION_CENTER);

    assert_eq!(selected.candidates, 2);
    assert_eq!(selected.inside_fov, 1);
    assert_eq!(selected.target_object_id, Some(2));
}

#[test]
fn control_aim_point_is_separate_from_association_center_and_supports_class_override() {
    let mut class_ratios = BTreeMap::new();
    class_ratios.insert(1, 0.304);
    let mut core = TargetingCore::new(TargetingConfig {
        aim_y_ratio: 0.22,
        class_aim_y_ratios: class_ratios,
        ..TargetingConfig::default()
    });
    let target = Detection::new(1, 1, 280.0, 200.0, 80.0, 100.0, 0.9).expect("target");
    let selection = core.select(&[target], OBSERVATION_CENTER);
    assert_eq!(selection.target_aim_x, Some(320.0));
    assert_eq!(selection.target_aim_y, Some(230.0));
    assert_eq!(selection.target_box_x, Some(280.0));
    assert_eq!(selection.target_box_y, Some(200.0));
    assert_eq!(selection.target_box_width, Some(80.0));
    assert_eq!(selection.target_box_height, Some(100.0));
    assert_eq!(core.locked().expect("track").center_y, 230.0);
}

#[test]
fn extreme_aspect_ratio_is_rejected_before_association() {
    let mut core = TargetingCore::new(TargetingConfig {
        candidate_max_aspect_ratio: 6.0,
        ..TargetingConfig::default()
    });
    let stretched = Detection::new(1, 0, 270.0, 315.0, 100.0, 10.0, 0.9).expect("stretched");
    let selection = core.select(&[stretched], OBSERVATION_CENTER);
    assert!(selection.target_object_id.is_none());
    assert_eq!(selection.inside_fov, 0);
}

#[test]
fn head_movement_keeps_the_same_stable_lock() {
    let mut core = TargetingCore::new(TargetingConfig::default());
    let frame1 = vec![Detection::new(1, 0, 300.0, 300.0, 40.0, 80.0, 0.9).expect("d1")];
    let first = core.select(&frame1, OBSERVATION_CENTER);
    assert_eq!(first.lock_reason, Some(LockReason::PreferredClass));
    // Head moved 20 px, still within debounce window.
    let frame2 = vec![Detection::new(1, 0, 320.0, 300.0, 40.0, 80.0, 0.9).expect("d2")];
    let second = core.select(&frame2, OBSERVATION_CENTER);
    assert_eq!(second.target_object_id, Some(1));
    assert_eq!(second.lock_reason, Some(LockReason::PreferredClass));
}

#[test]
fn stable_challenger_must_hold_its_advantage_for_capture_time_delay() {
    let mut core = TargetingCore::new(TargetingConfig {
        selection_class_weight: 0.01,
        selection_distance_weight: 0.99,
        sticky_bias: 0.0,
        switch_min_preference_advantage: 0.01,
        switch_delay_ms: 50.0,
        ..TargetingConfig::default()
    });
    let first = vec![
        Detection::new(1, 0, 270.0, 285.0, 80.0, 160.0, 0.9).expect("locked"),
        Detection::new(2, 1, 295.0, 285.0, 80.0, 160.0, 0.9).expect("challenger"),
    ];
    assert!(
        core.select_at(&first, OBSERVATION_CENTER, 1_000_000_000)
            .target_object_id
            .is_none()
    );
    assert_eq!(
        core.select_at(&first, OBSERVATION_CENTER, 1_001_000_000)
            .target_object_id,
        Some(1)
    );

    let challenger_wins = vec![
        Detection::new(11, 0, 288.0, 285.0, 80.0, 160.0, 0.9).expect("locked moved"),
        Detection::new(12, 1, 280.0, 285.0, 80.0, 160.0, 0.9).expect("stable challenger"),
    ];
    let pending = core.select_at(&challenger_wins, OBSERVATION_CENTER, 1_020_000_000);
    assert_eq!(pending.target_object_id, Some(11), "delay must hold lock");

    for timestamp in [1_040_000_000, 1_080_000_000, 1_120_000_000] {
        core.select_at(&challenger_wins, OBSERVATION_CENTER, timestamp);
    }
    let committed = core.select_at(&challenger_wins, OBSERVATION_CENTER, 1_180_000_000);
    assert_eq!(
        committed.target_object_id,
        Some(12),
        "capture-time advantage held beyond 50 ms must switch"
    );
    assert_eq!(committed.lock_reason, Some(LockReason::FallbackClass));
}

#[test]
fn frame_local_candidate_reordering_keeps_runtime_track_identity() {
    let mut core = TargetingCore::new(TargetingConfig::default());
    let left = Detection::new(0, 0, 280.0, 300.0, 40.0, 80.0, 0.9).expect("left");
    let right = Detection::new(1, 0, 500.0, 300.0, 40.0, 80.0, 0.9).expect("right");
    let first = core.select(&[left, right], OBSERVATION_CENTER);
    let stable_track = first.target_track_id.expect("track identity");

    // DeepStream changed list order, so the same spatial candidate now has
    // frame-local ID 1 while the other candidate has ID 0.
    let other = Detection::new(0, 0, 500.0, 300.0, 40.0, 80.0, 0.9).expect("other");
    let same = Detection::new(1, 0, 282.0, 300.0, 40.0, 80.0, 0.9).expect("same");
    let second = core.select(&[other, same], OBSERVATION_CENTER);

    assert_eq!(second.target_object_id, Some(1));
    assert_eq!(second.target_track_id, Some(stable_track));
    assert_eq!(core.locked().expect("locked").age_frames, 2);
}

#[test]
fn identity_confidence_is_spatial_continuity_not_detector_confidence() {
    let mut core = TargetingCore::new(TargetingConfig::default());
    let first = Detection::new(1, 0, 300.0, 300.0, 40.0, 80.0, 0.55).expect("first");
    assert!(
        core.select(std::slice::from_ref(&first), OBSERVATION_CENTER)
            .target_track_id
            .is_none()
    );
    core.select(&[first], OBSERVATION_CENTER);
    let acquired = core.locked().expect("acquired track");
    assert!((acquired.confidence - 0.55).abs() < f32::EPSILON);
    assert_eq!(acquired.identity_confidence, 1.0);

    let moved = Detection::new(2, 0, 318.0, 300.0, 40.0, 80.0, 0.95).expect("moved");
    let selection = core.select(&[moved], OBSERVATION_CENTER);
    let tracked = core.locked().expect("continued track");
    let expected = 1.0 - (0.75 * 0.225 + 0.25 * (1.0 - 22.0 / 58.0)) / 1.15;
    assert!((tracked.confidence - 0.95).abs() < f32::EPSILON);
    assert!((tracked.identity_confidence - expected).abs() < 1e-12);
    assert_eq!(selection.target_detection_confidence, Some(0.95));
    assert!(
        (selection
            .target_identity_confidence
            .expect("selection identity confidence")
            - expected)
            .abs()
            < 1e-12
    );
}

#[test]
fn rectangular_minimum_cost_assignment_preserves_both_feasible_identities() {
    let mut core = TargetingCore::new(TargetingConfig {
        target_fov_radius_px: 200.0,
        ..TargetingConfig::default()
    });
    let first = vec![
        Detection::new(1, 0, 280.0, 270.0, 40.0, 100.0, 0.9).expect("left"),
        Detection::new(2, 0, 300.0, 270.0, 40.0, 100.0, 0.9).expect("locked"),
    ];
    assert!(
        core.select(&first, OBSERVATION_CENTER)
            .target_track_id
            .is_none()
    );
    let locked_id = core
        .select(&first, OBSERVATION_CENTER)
        .target_track_id
        .expect("locked id");
    let squeezed = vec![
        Detection::new(11, 0, 290.0, 270.0, 40.0, 100.0, 0.9).expect("left moved"),
        Detection::new(12, 0, 318.0, 270.0, 40.0, 100.0, 0.9).expect("locked moved"),
    ];
    let selection = core.select(&squeezed, OBSERVATION_CENTER);
    assert_eq!(selection.target_track_id, Some(locked_id));
    assert_eq!(
        selection.target_object_id,
        Some(12),
        "minimum-cost assignment must not let the locked track steal detection 11"
    );
}

#[test]
fn scale_jump_and_expired_capture_gap_allocate_new_identities() {
    let mut scale_core = TargetingCore::new(TargetingConfig {
        tracker_max_size_ratio: 2.5,
        ..TargetingConfig::default()
    });
    let first = Detection::new(1, 0, 280.0, 270.0, 80.0, 100.0, 0.9).expect("first");
    let first_id = scale_core
        .select_at(&[first], OBSERVATION_CENTER, 1_000_000_000)
        .target_track_id
        .expect("first id");
    let resized = Detection::new(2, 0, 300.0, 300.0, 20.0, 25.0, 0.9).expect("resized");
    assert_ne!(
        scale_core
            .select_at(&[resized], OBSERVATION_CENTER, 1_010_000_000)
            .target_track_id,
        Some(first_id)
    );

    let mut time_core = TargetingCore::new(TargetingConfig {
        tracker_max_association_dt_ms: 150.0,
        ..TargetingConfig::default()
    });
    let first = Detection::new(1, 0, 280.0, 270.0, 80.0, 100.0, 0.9).expect("first");
    let first_id = time_core
        .select_at(&[first], OBSERVATION_CENTER, 2_000_000_000)
        .target_track_id
        .expect("first id");
    let same = Detection::new(2, 0, 280.0, 270.0, 80.0, 100.0, 0.9).expect("same");
    assert_ne!(
        time_core
            .select_at(&[same], OBSERVATION_CENTER, 2_151_000_000)
            .target_track_id,
        Some(first_id)
    );
}

#[test]
fn missing_locked_target_does_not_promote_a_different_class_during_grace() {
    let mut core = TargetingCore::new(TargetingConfig {
        track_max_age: 2,
        ..TargetingConfig::default()
    });
    let first = vec![
        Detection::new(1, 0, 280.0, 270.0, 80.0, 100.0, 0.9).expect("locked"),
        Detection::new(2, 1, 380.0, 270.0, 80.0, 100.0, 0.9).expect("challenger"),
    ];
    core.select(&first, OBSERVATION_CENTER);
    let locked_only = Detection::new(11, 0, 280.0, 270.0, 80.0, 100.0, 0.9).expect("locked");
    core.select(&[locked_only], OBSERVATION_CENTER);
    let challenger =
        Detection::new(12, 1, 380.0, 270.0, 80.0, 100.0, 0.9).expect("challenger returns");
    let selection = core.select(&[challenger], OBSERVATION_CENTER);
    assert_eq!(selection.target_track_id, None);
    assert_eq!(
        core.locked().map(|track| track.id),
        Some(novasight_core::tracking::TrackId(1))
    );
}

#[test]
fn temporarily_missing_locked_target_does_not_immediately_switch_to_challenger() {
    let mut core = TargetingCore::new(TargetingConfig {
        track_max_age: 2,
        track_max_lost_age_ms: 120.0,
        ..TargetingConfig::default()
    });
    let both_visible = [
        Detection::new(1, 0, 280.0, 270.0, 80.0, 100.0, 0.9).expect("locked"),
        Detection::new(2, 0, 380.0, 270.0, 80.0, 100.0, 0.9).expect("challenger"),
    ];
    assert!(
        core.select_at(&both_visible, OBSERVATION_CENTER, 1_000_000_000)
            .target_track_id
            .is_none()
    );
    let locked = core.select_at(&both_visible, OBSERVATION_CENTER, 1_010_000_000);
    let locked_track = locked.target_track_id.expect("confirmed lock");

    let challenger_only =
        Detection::new(12, 0, 380.0, 270.0, 80.0, 100.0, 0.9).expect("challenger only");
    let missing = core.select_at(&[challenger_only], OBSERVATION_CENTER, 1_020_000_000);

    assert_eq!(
        missing.target_track_id, None,
        "one missed observation must pause output instead of changing targets"
    );
    assert_eq!(
        core.locked().map(|track| track.id),
        Some(locked_track),
        "the lost-track grace period must retain lock ownership"
    );

    let both_returned = [
        Detection::new(21, 0, 282.0, 270.0, 80.0, 100.0, 0.9).expect("locked returned"),
        Detection::new(22, 0, 380.0, 270.0, 80.0, 100.0, 0.9).expect("challenger"),
    ];
    let reacquired = core.select_at(&both_returned, OBSERVATION_CENTER, 1_030_000_000);
    assert_eq!(reacquired.target_track_id, Some(locked_track));
    assert_eq!(reacquired.target_object_id, Some(21));
    assert!(reacquired.target_rebuilt);
}

#[test]
fn locked_target_loss_grace_uses_capture_time_instead_of_inference_frame_count() {
    let mut core = TargetingCore::new(TargetingConfig {
        track_max_age: 2,
        track_max_lost_age_ms: 120.0,
        ..TargetingConfig::default()
    });
    let both_visible = [
        Detection::new(1, 0, 280.0, 270.0, 80.0, 100.0, 0.9).expect("locked"),
        Detection::new(2, 0, 380.0, 270.0, 80.0, 100.0, 0.9).expect("challenger"),
    ];
    core.select_at(&both_visible, OBSERVATION_CENTER, 1_000_000_000);
    let locked = core.select_at(&both_visible, OBSERVATION_CENTER, 1_008_333_333);
    let locked_track = locked.target_track_id.expect("confirmed lock");
    let challenger_only =
        Detection::new(12, 0, 380.0, 270.0, 80.0, 100.0, 0.9).expect("challenger only");

    for timestamp in [1_016_666_666, 1_024_999_999, 1_033_333_332, 1_041_666_665] {
        let selection = core.select_at(
            std::slice::from_ref(&challenger_only),
            OBSERVATION_CENTER,
            timestamp,
        );
        assert_eq!(selection.target_track_id, None);
        assert_eq!(core.locked().map(|track| track.id), Some(locked_track));
    }

    let expired = core.select_at(&[challenger_only], OBSERVATION_CENTER, 1_140_000_000);
    assert_ne!(expired.target_track_id, Some(locked_track));
    assert!(expired.target_track_id.is_some());
}

#[test]
fn association_beyond_normalized_distance_allocates_a_new_identity() {
    let mut core = TargetingCore::new(TargetingConfig {
        target_fov_radius_px: 1_000.0,
        ..TargetingConfig::default()
    });
    let first = Detection::new(1, 0, 100.0, 100.0, 40.0, 80.0, 0.9).expect("first");
    let first_track = core
        .select(&[first], OBSERVATION_CENTER)
        .target_track_id
        .expect("first identity");
    let jumped = Detection::new(2, 0, 221.0, 100.0, 40.0, 80.0, 0.9).expect("jumped");
    let next = core.select(&[jumped], OBSERVATION_CENTER);
    assert_ne!(next.target_track_id, Some(first_track));
    assert_eq!(core.locked().expect("new track").identity_confidence, 1.0);
}

#[test]
fn filtered_classes_age_out_the_previous_track() {
    let mut core = TargetingCore::new(TargetingConfig {
        track_max_age: 2,
        allowed_class_ids: Some(BTreeSet::from([0, 1])),
        ..TargetingConfig::default()
    });
    let target = Detection::new(0, 0, 280.0, 300.0, 40.0, 80.0, 0.9).expect("target");
    let first = core.select(std::slice::from_ref(&target), OBSERVATION_CENTER);
    let first_track = first.target_track_id.expect("first track");
    let irrelevant = Detection::new(0, 2, 280.0, 300.0, 40.0, 80.0, 0.9).expect("irrelevant");
    for _ in 0..3 {
        assert!(
            core.select(std::slice::from_ref(&irrelevant), OBSERVATION_CENTER)
                .target_track_id
                .is_none()
        );
    }

    let reacquired = core.select(&[target], OBSERVATION_CENTER);
    assert_ne!(reacquired.target_track_id, Some(first_track));
}

#[test]
fn history_is_bounded() {
    let mut core = TargetingCore::new(TargetingConfig::default());
    for index in 0..(novasight_core::tracking::DEFAULT_HISTORY_LIMIT * 4) {
        let x = 300.0 + index as f32;
        let detection = Detection::new(1, 0, x, 300.0, 40.0, 80.0, 0.9).expect("d");
        core.select(&[detection], OBSERVATION_CENTER);
    }
    let count = core.history_iter().count();
    assert!(count <= novasight_core::tracking::DEFAULT_HISTORY_LIMIT);
}

#[test]
fn lost_track_cannot_produce_a_target_object_id() {
    let mut core = TargetingCore::new(TargetingConfig::default());
    let head = Detection::new(1, 0, 300.0, 280.0, 40.0, 80.0, 0.9).expect("head");
    core.select(&[head], OBSERVATION_CENTER);
    // Two empty frames in a row force the lock to Lost; the next
    // admissible frame restarts the targeting state cleanly.
    core.select(&[], OBSERVATION_CENTER);
    core.select(&[], OBSERVATION_CENTER);
    let body = Detection::new(2, 1, 320.0, 360.0, 30.0, 60.0, 0.9).expect("body");
    let next = core.select(&[body], OBSERVATION_CENTER);
    assert_eq!(next.target_object_id, Some(2));
    assert_eq!(next.target_class_id, Some(1));
}
