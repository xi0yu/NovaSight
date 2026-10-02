//! Tracker and targeting contract tests. Pins the deterministic
//! selection and reset edges from `target-switch-loss.jsonl` and a few
//! negative cases. The fixtures preserve the target-selection edges the
//! runtime must continue to respect.
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use novasight_core::perception::types::Detection;
use novasight_core::tracking::{
    KalmanConfig, LockReason, SelectionWeights, TargetPartRole, TargetingConfig, TargetingCore,
};
use serde_json::Value;

const OBSERVATION_CENTER: (f64, f64) = (320.0, 320.0);

#[test]
fn default_scoring_handles_distance_class_conflicts_reordering_and_short_loss() {
    // Declared preferences, not a search for a universal optimum. Exercise the
    // real selector together with identity retention and switching hysteresis.
    for scale in [0.5, 1.0, 2.0] {
        for fps in [30, 60, 120, 240] {
            for (classes, distances, confidence, expected) in [
                ([0, 0], [40.0, 80.0], [0.9, 0.9], 10),
                ([0, 1], [100.0, 40.0], [0.9, 0.9], 10),
                ([0, 1], [160.0, 20.0], [0.9, 0.9], 20),
                ([2, 2], [40.0, 40.0], [0.8, 0.95], 20),
            ] {
                let center = (320.0 * scale, 320.0 * scale);
                let mut core = TargetingCore::new(TargetingConfig {
                    target_fov_radius_px: 180.0 * scale,
                    aim_y_ratio: 0.5,
                    ..TargetingConfig::default()
                });
                let mut last = Vec::new();
                let mut identity = None;
                for frame in 0..12 {
                    let jitter = if frame % 2 == 0 { 0.5 } else { -0.5 };
                    let mut detections: Vec<_> = (0..2)
                        .map(|i| {
                            let x = center.0
                                + (if i == 0 { -1.0 } else { 1.0 })
                                    * (distances[i] + jitter)
                                    * scale;
                            Detection::new(
                                (i as u64 + 1) * 10,
                                classes[i],
                                (x - 10.0 * scale) as f32,
                                (center.1 - 20.0 * scale) as f32,
                                (20.0 * scale) as f32,
                                (40.0 * scale) as f32,
                                confidence[i],
                            )
                            .unwrap()
                        })
                        .collect();
                    if frame % 2 == 1 {
                        detections.reverse();
                    }
                    let result = core.select_at(
                        &detections,
                        center,
                        1_000_000_000 + frame * 1_000_000_000 / fps,
                    );
                    if frame > 0 {
                        assert_eq!(
                            result.target_object_id,
                            Some(expected),
                            "scale={scale} fps={fps} frame={frame}"
                        );
                        if let Some(id) = identity {
                            assert_eq!(result.target_track_id, Some(id));
                        }
                        identity = result.target_track_id;
                    }
                    last = detections;
                }
                assert!(
                    core.select_at(&[], center, 1_000_000_000 + 12 * 1_000_000_000 / fps)
                        .target_object_id
                        .is_none()
                );
                let recovered =
                    core.select_at(&last, center, 1_000_000_000 + 13 * 1_000_000_000 / fps);
                assert_eq!(recovered.target_track_id, identity);
                assert_eq!(recovered.target_object_id, Some(expected));
            }
        }
    }
}

#[test]
fn explicit_class_weights_and_xy_points_reach_target_selection() {
    let mut config = TargetingConfig {
        class_weights: BTreeMap::from([(0, 0.8), (1, 0.4)]),
        class_aim_x_ratios: BTreeMap::from([(0, 0.25), (1, 0.75)]),
        class_aim_y_ratios: BTreeMap::from([(0, 0.4), (1, 0.6)]),
        selection_weights: SelectionWeights {
            distance: 0.0,
            class: 1.0,
            confidence: 0.0,
        },
        switch_delay_ms: 0.0,
        ..TargetingConfig::default()
    };
    let detections = [
        Detection::new(10, 0, 300.0, 280.0, 40.0, 80.0, 0.9).unwrap(),
        Detection::new(20, 1, 300.0, 280.0, 40.0, 80.0, 0.9).unwrap(),
    ];
    let mut core = TargetingCore::new(config.clone());
    core.select_at(&detections, OBSERVATION_CENTER, 1_000_000_000);
    let first = core.select_at(&detections, OBSERVATION_CENTER, 1_020_000_000);
    assert_eq!(first.target_object_id, Some(10));
    assert_eq!(
        (first.target_aim_x, first.target_aim_y),
        (Some(310.0), Some(312.0))
    );
    config.class_weights = BTreeMap::from([(0, 0.4), (1, 0.8)]);
    let mut core = TargetingCore::new(config);
    core.select_at(&detections, OBSERVATION_CENTER, 1_000_000_000);
    let second = core.select_at(&detections, OBSERVATION_CENTER, 1_020_000_000);
    assert_eq!(second.target_object_id, Some(20));
    assert_eq!(
        (second.target_aim_x, second.target_aim_y),
        (Some(330.0), Some(328.0))
    );
    assert_eq!(second.lock_reason, Some(LockReason::NearbyAim));
}

#[test]
fn search_circle_uses_observed_aim_inclusive_boundary_and_not_box_size() {
    let cases = [
        // All aim points are (320,320). A small distant box is still eligible.
        (10.0, 20.0, 100.0, (420.0, 320.0), true),
        (40.0, 80.0, 100.0, (420.0, 320.0), true),
        (80.0, 40.0, 100.0, (320.0, 220.0), true),
        (40.0, 80.0, 100.0, (380.0, 400.0), true),
        (40.0, 80.0, 100.0, (420.1, 320.0), false),
        // Even a box covering the crosshair cannot admit an out-of-range aim.
        (240.0, 240.0, 100.0, (430.0, 320.0), false),
        (40.0, 80.0, f64::NAN, (320.0, 320.0), false),
        (40.0, 80.0, f64::INFINITY, (320.0, 320.0), false),
        (40.0, 80.0, 0.0, (320.0, 320.0), false),
        (40.0, 80.0, -1.0, (320.0, 320.0), false),
    ];
    for (width, height, radius, point, expected) in cases {
        let mut core = TargetingCore::new(TargetingConfig {
            target_fov_radius_px: radius,
            aim_y_ratio: 0.5,
            ..TargetingConfig::default()
        });
        let target = Detection::new(
            1,
            0,
            320.0 - width / 2.0,
            320.0 - height / 2.0,
            width,
            height,
            0.95,
        )
        .unwrap();
        let result = core.select(&[target], point);
        assert_eq!(
            result.target_object_id.is_some(),
            expected,
            "{width}x{height} radius={radius} point={point:?}"
        );
        assert_eq!(result.inside_fov, usize::from(expected));
    }
    for (aim_x_ratio, admitted) in [(0.5, true), (1.0, false)] {
        let mut core = TargetingCore::new(TargetingConfig {
            target_fov_radius_px: 100.0,
            class_aim_x_ratios: BTreeMap::from([(0, aim_x_ratio)]),
            aim_y_ratio: 0.5,
            ..TargetingConfig::default()
        });
        let target = Detection::new(1, 0, 300.0, 280.0, 200.0, 80.0, 0.95).unwrap();
        assert_eq!(
            core.select(&[target], OBSERVATION_CENTER)
                .target_object_id
                .is_some(),
            admitted
        );
    }
}

#[test]
fn search_circle_exit_suppresses_output_without_discarding_observed_identity() {
    let mut config = TargetingConfig {
        target_fov_radius_px: 20.0,
        aim_y_ratio: 0.5,
        ..TargetingConfig::default()
    };
    let mut core = TargetingCore::new(config.clone());
    let target = Detection::new(1, 0, 300.0, 280.0, 40.0, 80.0, 0.95).unwrap();
    let id = core
        .select_at(&[target.clone()], (320.0, 320.0), 1_000_000_000)
        .target_track_id
        .unwrap();
    let outside = core.select_at(&[target.clone()], (345.0, 320.0), 1_010_000_000);
    assert!(outside.target_object_id.is_none());
    assert_eq!(core.locked().unwrap().id, id);
    // A live radius change affects the next observation, retaining identity.
    config.target_fov_radius_px = 30.0;
    core.set_config(config.clone());
    let inside = core.select_at(&[target.clone()], (345.0, 320.0), 1_020_000_000);
    assert_eq!(inside.target_track_id, Some(id));
    assert!(!inside.target_rebuilt);
    config.target_fov_radius_px = 10.0;
    core.set_config(config);
    assert!(
        core.select_at(&[target], (345.0, 320.0), 1_030_000_000)
            .target_object_id
            .is_none()
    );
    assert!(
        core.select_at(&[], (320.0, 320.0), 1_040_000_000)
            .target_object_id
            .is_none()
    );
}

#[test]
fn search_circle_skips_outside_candidates_and_uses_latest_observed_aim() {
    let mut core = TargetingCore::new(TargetingConfig {
        target_fov_radius_px: 20.0,
        aim_y_ratio: 0.5,
        ..TargetingConfig::default()
    });
    let outside = Detection::new(1, 0, 350.0, 280.0, 40.0, 80.0, 0.99).unwrap();
    let inside = Detection::new(2, 0, 300.0, 280.0, 40.0, 80.0, 0.95).unwrap();
    let selection = core.select_at(&[outside, inside], OBSERVATION_CENTER, 1_000_000_000);
    assert_eq!(selection.target_object_id, Some(2));
    assert_eq!(selection.rejected_by_fov, 1);
    let moved = Detection::new(3, 0, 321.0, 280.0, 40.0, 80.0, 0.95).unwrap();
    assert!(
        core.select_at(&[moved], OBSERVATION_CENTER, 1_010_000_000)
            .target_object_id
            .is_none()
    );
}

#[test]
fn search_radius_does_not_change_ranking_or_switching_for_admitted_targets() {
    let near = Detection::new(1, 1, 310.0, 280.0, 40.0, 80.0, 0.95).unwrap();
    let preferred = Detection::new(2, 0, 400.0, 280.0, 40.0, 80.0, 0.95).unwrap();
    let mut traces = Vec::new();
    for radius in [120.0, 1000.0] {
        let config = TargetingConfig {
            target_fov_radius_px: radius,
            aim_y_ratio: 0.5,
            switch_min_preference_advantage: 0.005,
            switch_delay_ms: 30.0,
            ..TargetingConfig::default()
        };
        let mut core = TargetingCore::new(config.clone());
        let mut trace = vec![
            core.select_at(&[near.clone()], OBSERVATION_CENTER, 1_000_000_000)
                .target_object_id,
        ];
        for frame in 1..=6 {
            let result = core.select_at(
                &[near.clone(), preferred.clone()],
                OBSERVATION_CENTER,
                1_000_000_000 + frame * 20_000_000,
            );
            trace.push(result.target_object_id);
        }
        assert_eq!(trace.last(), Some(&Some(1)));
        traces.push(trace);

        let mut fresh = TargetingCore::new(config);
        for frame in 0..=1 {
            let selected = fresh.select_at(
                &[near.clone(), preferred.clone()],
                OBSERVATION_CENTER,
                2_000_000_000 + frame * 20_000_000,
            );
            if frame == 1 {
                assert_eq!(selected.target_object_id, Some(1));
            }
        }
    }
    assert_eq!(traces[0], traces[1]);
}

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
        "nearby_aim" => LockReason::NearbyAim,
        "maintained_target" => LockReason::MaintainedTarget,

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
    assert_eq!(
        core.select_at(&first_detections, OBSERVATION_CENTER, first_captured_at_ns)
            .target_object_id,
        None,
        "both candidates are in the circle; ambiguous first observations need confirmation"
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
    let mut core = TargetingCore::new(TargetingConfig::default());
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
fn strong_geometry_can_keep_identity_when_raw_class_changes() {
    let mut core = TargetingCore::new(TargetingConfig::default());
    let first = Detection::new(1, 0, 280.0, 280.0, 80.0, 100.0, 0.95).expect("first");
    let track_id = core
        .select_at(&[first], OBSERVATION_CENTER, 1_000_000_000)
        .target_track_id
        .expect("initial track");

    let changed_class =
        Detection::new(2, 1, 282.0, 280.0, 80.0, 100.0, 0.95).expect("changed class");
    let selection = core.select_at(&[changed_class], OBSERVATION_CENTER, 1_010_000_000);

    assert_eq!(selection.target_track_id, Some(track_id));
    assert_eq!(selection.target_class_id, Some(1));
    assert!(!selection.target_rebuilt);
}

#[test]
fn same_class_association_wins_over_a_slightly_closer_cross_class_box() {
    let mut core = TargetingCore::new(TargetingConfig::default());
    let first = Detection::new(1, 0, 280.0, 280.0, 80.0, 100.0, 0.95).expect("first");
    let track_id = core
        .select_at(&[first], OBSERVATION_CENTER, 1_000_000_000)
        .target_track_id
        .expect("initial track");
    let candidates = [
        Detection::new(2, 1, 280.0, 280.0, 80.0, 100.0, 0.95).expect("cross class"),
        Detection::new(3, 0, 290.0, 280.0, 80.0, 100.0, 0.95).expect("same class"),
    ];

    let selection = core.select_at(&candidates, OBSERVATION_CENTER, 1_010_000_000);

    assert_eq!(selection.target_track_id, Some(track_id));
    assert_eq!(selection.target_object_id, Some(3));
    assert_eq!(selection.target_class_id, Some(0));
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
        let track_id = selected.target_track_id.unwrap_or_else(|| {
            panic!(
                "frame {frame} lost its smoothly moving target: {selected:?}; lock={:?}",
                core.locked()
            )
        });
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
fn abrupt_jump_after_smooth_motion_cannot_reuse_identity_via_observed_position_fallback() {
    let mut core = TargetingCore::new(TargetingConfig::default());
    let mut stable_track = None;
    for (frame, x) in [240.0_f32, 256.0, 272.0].into_iter().enumerate() {
        let detection = Detection::new(frame as u64, 0, x, 240.0, 80.0, 160.0, 0.95).unwrap();
        let selected = core.select_at(
            &[detection],
            OBSERVATION_CENTER,
            1_000_000_000 + frame as u64 * 8_333_333,
        );
        let track = selected
            .target_track_id
            .expect("smooth motion remains associated");
        if let Some(expected) = stable_track {
            assert_eq!(track, expected);
        }
        stable_track = Some(track);
    }

    // The 80px jump remains inside both the search circle and the height-normalized
    // distance bound. Only the NIS outlier check rejects both reference points.
    let jumped = Detection::new(10, 0, 352.0, 240.0, 80.0, 160.0, 0.95).unwrap();
    let rejected = core.select_at(
        std::slice::from_ref(&jumped),
        OBSERVATION_CENTER,
        1_025_000_000,
    );
    assert_eq!(rejected.inside_fov, 1);
    assert!(rejected.target_track_id.is_none());
    assert!(rejected.target_aim_x.is_none());
    assert!(rejected.target_aim_y.is_none());
    assert_eq!(core.locked().map(|track| track.id), stable_track);

    // Repeated observations may confirm the new target, never recycle the
    // old identity merely because the fallback hypothesis was available.
    let waiting = core.select_at(
        std::slice::from_ref(&jumped),
        OBSERVATION_CENTER,
        1_035_000_000,
    );
    assert!(waiting.target_track_id.is_none());
    let confirmed = core.select_at(&[jumped], OBSERVATION_CENTER, 1_095_000_000);
    assert!(confirmed.target_track_id.is_some());
    assert_ne!(confirmed.target_track_id, stable_track);
    assert_eq!(confirmed.target_object_id, Some(10));
}

#[test]
fn valid_lock_is_not_replaced_by_a_higher_merit_candidate() {
    let mut core = TargetingCore::new(TargetingConfig {
        class_weights: BTreeMap::new(),
        selection_weights: SelectionWeights {
            distance: 1.0,
            class: 0.0,
            confidence: 0.0,
        },
        switch_min_continuity_score: 0.0,
        switch_delay_ms: 0.0,
        kalman: KalmanConfig {
            nis_threshold: 1_000_000.0,
            nis_hard_reject: 1_000_000.0,
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
}

#[test]
fn equal_scores_tie_break_by_stable_track_id_not_frame_local_object_id() {
    let mut core = TargetingCore::new(TargetingConfig {
        class_weights: BTreeMap::new(),
        selection_weights: SelectionWeights {
            distance: 1.0,
            class: 0.0,
            confidence: 0.0,
        },
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
fn reset_clears_targeting_state() {
    let mut core = TargetingCore::new(TargetingConfig::default());
    let head = Detection::new(1, 0, 300.0, 280.0, 40.0, 80.0, 0.9).expect("head");
    core.select(&[head], OBSERVATION_CENTER);
    core.reset();
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
fn explicit_class_filter_is_independent_from_class_weights() {
    let mut core = TargetingCore::new(TargetingConfig {
        class_weights: BTreeMap::from([(0, 1.0), (1, 0.5)]),
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
    assert_eq!(selected.inside_fov, 2);
    assert!(
        selected.target_object_id.is_none(),
        "two new candidates need confirmation"
    );
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
fn detections_outside_selection_fov_keep_their_tracker_identity() {
    let mut core = TargetingCore::new(TargetingConfig {
        target_fov_radius_px: 20.0,
        kalman: KalmanConfig {
            nis_threshold: 1_000_000.0,
            nis_hard_reject: 1_000_000.0,
            ..KalmanConfig::default()
        },
        ..TargetingConfig::default()
    });
    let initial = Detection::new(1, 0, 300.0, 300.0, 40.0, 80.0, 0.95).expect("initial");
    let track_id = core
        .select_at(&[initial], OBSERVATION_CENTER, 1_000_000_000)
        .target_track_id
        .expect("initial lock");

    let outside = Detection::new(2, 0, 322.0, 300.0, 40.0, 80.0, 0.95).expect("outside");
    let suppressed = core.select_at(&[outside], OBSERVATION_CENTER, 1_010_000_000);
    assert!(suppressed.target_track_id.is_none());
    assert_eq!(suppressed.rejected_by_fov, 1);
    assert_eq!(
        core.locked().expect("tracked outside selection fov").id,
        track_id
    );

    let returned = Detection::new(3, 0, 302.0, 300.0, 40.0, 80.0, 0.95).expect("returned");
    let reacquired = core.select_at(&[returned], OBSERVATION_CENTER, 1_020_000_000);
    assert_eq!(reacquired.target_track_id, Some(track_id));
    assert!(!reacquired.target_rebuilt);
}

#[test]
fn every_declared_class_weight_affects_selection() {
    let mut core = TargetingCore::new(TargetingConfig {
        class_weights: BTreeMap::from([(0, 1.0), (1, 0.5), (2, 0.25), (3, 0.125)]),
        selection_weights: SelectionWeights {
            distance: 0.0,
            class: 1.0,
            confidence: 0.0,
        },
        ..TargetingConfig::default()
    });
    let candidates = [
        Detection::new(1, 2, 280.0, 280.0, 40.0, 80.0, 0.95).expect("third rank"),
        Detection::new(2, 3, 340.0, 280.0, 40.0, 80.0, 0.95).expect("fourth rank"),
    ];
    assert!(
        core.select_at(&candidates, OBSERVATION_CENTER, 1_000_000_000)
            .target_track_id
            .is_none()
    );
    let selected = core.select_at(&candidates, OBSERVATION_CENTER, 1_010_000_000);
    assert_eq!(selected.target_class_id, Some(2));
}

#[test]
fn unweighted_classes_receive_no_implicit_preference() {
    let mut core = TargetingCore::new(TargetingConfig {
        class_weights: BTreeMap::new(),
        selection_weights: SelectionWeights {
            distance: 0.0,
            class: 1.0,
            confidence: 0.0,
        },
        ..TargetingConfig::default()
    });
    let candidates = [
        Detection::new(1, 15, 300.0, 280.0, 40.0, 80.0, 0.95).unwrap(),
        Detection::new(2, 0, 300.0, 280.0, 40.0, 80.0, 0.95).unwrap(),
    ];
    core.select_at(&candidates, OBSERVATION_CENTER, 1_000_000_000);
    let selected = core.select_at(&candidates, OBSERVATION_CENTER, 1_010_000_000);

    assert_eq!(selected.target_object_id, Some(1));
    assert_eq!(selected.target_class_id, Some(15));
}

#[test]
fn confidence_can_outweigh_a_small_distance_advantage() {
    let mut core = TargetingCore::new(TargetingConfig {
        class_weights: BTreeMap::new(),
        selection_weights: SelectionWeights {
            distance: 0.65,
            class: 0.0,
            confidence: 0.15,
        },
        ..TargetingConfig::default()
    });
    let candidates = [
        Detection::new(1, 0, 300.0, 280.0, 40.0, 80.0, 0.51).expect("near low score"),
        Detection::new(2, 0, 315.0, 280.0, 40.0, 80.0, 0.99).expect("far high score"),
    ];
    core.select_at(&candidates, OBSERVATION_CENTER, 1_000_000_000);
    let selected = core.select_at(&candidates, OBSERVATION_CENTER, 1_010_000_000);

    assert_eq!(selected.target_object_id, Some(2));
}

#[test]
fn box_size_does_not_break_a_tie_with_the_same_observed_aim_point() {
    let mut core = TargetingCore::new(TargetingConfig {
        class_weights: BTreeMap::new(),
        aim_y_ratio: 0.5,
        ..TargetingConfig::default()
    });
    let candidates = [
        Detection::new(1, 0, 310.0, 300.0, 20.0, 40.0, 0.9).expect("small"),
        Detection::new(2, 0, 300.0, 280.0, 40.0, 80.0, 0.9).expect("large"),
    ];
    core.select_at(&candidates, OBSERVATION_CENTER, 1_000_000_000);
    let selected = core.select_at(&candidates, OBSERVATION_CENTER, 1_010_000_000);

    assert_eq!(selected.target_object_id, Some(1));
    assert_eq!(
        (selected.target_aim_x, selected.target_aim_y),
        (Some(320.0), Some(320.0))
    );
}

#[test]
fn association_continuity_does_not_add_target_selection_merit() {
    let initial_config = TargetingConfig {
        class_weights: BTreeMap::new(),
        aim_y_ratio: 0.5,
        switch_min_preference_advantage: 1.0,
        switch_min_continuity_score: 0.0,
        switch_delay_ms: 0.0,
        kalman: KalmanConfig {
            nis_threshold: 1_000_000.0,
            nis_hard_reject: 1_000_000.0,
            ..KalmanConfig::default()
        },
        ..TargetingConfig::default()
    };
    let mut core = TargetingCore::new(initial_config.clone());
    let initial = [
        Detection::new(1, 0, 290.0, 280.0, 40.0, 80.0, 0.9).expect("left"),
        Detection::new(2, 0, 330.0, 280.0, 40.0, 80.0, 0.9).expect("right"),
    ];
    core.select_at(&initial, OBSERVATION_CENTER, 1_000_000_000);
    let locked = core
        .select_at(&initial, OBSERVATION_CENTER, 1_010_000_000)
        .target_track_id
        .expect("locked");
    let displaced = [
        Detection::new(11, 0, 270.0, 280.0, 40.0, 80.0, 0.9).expect("left displaced"),
        Detection::new(12, 0, 330.0, 280.0, 40.0, 80.0, 0.9).expect("right stable"),
    ];
    core.set_config(TargetingConfig {
        switch_min_preference_advantage: 0.01,
        ..initial_config
    });
    let selected = core.select_at(&displaced, OBSERVATION_CENTER, 1_020_000_000);

    assert_eq!(selected.target_track_id, Some(locked));
    assert_eq!(selected.target_object_id, Some(11));
    assert!(!selected.target_rebuilt);
}

#[test]
fn motion_history_does_not_break_an_equal_current_distance_tie() {
    let mut core = TargetingCore::new(TargetingConfig {
        class_weights: BTreeMap::new(),
        aim_y_ratio: 0.5,
        switch_min_preference_advantage: 0.0,
        switch_min_continuity_score: 0.0,
        switch_delay_ms: 0.0,
        kalman: KalmanConfig {
            nis_threshold: 1_000_000.0,
            nis_hard_reject: 1_000_000.0,
            ..KalmanConfig::default()
        },
        ..TargetingConfig::default()
    });
    let initial = [
        Detection::new(1, 0, 280.0, 280.0, 40.0, 80.0, 0.9).expect("near left"),
        Detection::new(2, 0, 360.0, 280.0, 40.0, 80.0, 0.9).expect("far right"),
    ];
    core.select_at(&initial, OBSERVATION_CENTER, 1_000_000_000);
    let locked = core
        .select_at(&initial, OBSERVATION_CENTER, 1_010_000_000)
        .target_track_id
        .expect("locked");
    let moving = [
        Detection::new(11, 0, 260.0, 280.0, 40.0, 80.0, 0.9).expect("moving away"),
        Detection::new(12, 0, 340.0, 280.0, 40.0, 80.0, 0.9).expect("moving toward"),
    ];

    let selected = core.select_at(&moving, OBSERVATION_CENTER, 1_020_000_000);

    assert_eq!(selected.target_track_id, Some(locked));
    assert_eq!(selected.target_object_id, Some(11));
    assert!(!selected.target_rebuilt);
    let missing = core.select_at(&[], OBSERVATION_CENTER, 1_030_000_000);
    assert!(missing.target_object_id.is_none());
    assert!(missing.target_aim_x.is_none());
    assert!(missing.target_aim_y.is_none());
    assert_eq!(core.locked().expect("retained identity").id, locked);
}

#[test]
fn invalid_prediction_does_not_prevent_selecting_the_current_observation() {
    let mut core = TargetingCore::new(TargetingConfig {
        class_weights: BTreeMap::new(),
        aim_y_ratio: 0.5,
        selection_weights: SelectionWeights {
            distance: 1.0,
            class: 0.0,
            confidence: 0.0,
        },
        switch_min_preference_advantage: 0.0,
        switch_min_continuity_score: 0.0,
        switch_delay_ms: 0.0,
        kalman: KalmanConfig {
            min_identity_confidence: 1.1,
            nis_threshold: 1_000_000.0,
            nis_hard_reject: 1_000_000.0,
            ..KalmanConfig::default()
        },
        ..TargetingConfig::default()
    });
    let initial = [
        Detection::new(1, 0, 280.0, 280.0, 40.0, 80.0, 0.9).expect("left"),
        Detection::new(2, 0, 360.0, 280.0, 40.0, 80.0, 0.9).expect("right"),
    ];
    core.select_at(&initial, OBSERVATION_CENTER, 1_000_000_000);
    let locked = core
        .select_at(&initial, OBSERVATION_CENTER, 1_010_000_000)
        .target_track_id
        .expect("stable tie lock");
    let moving = [
        Detection::new(11, 0, 260.0, 280.0, 40.0, 80.0, 0.9).expect("moving away"),
        Detection::new(12, 0, 340.0, 280.0, 40.0, 80.0, 0.9).expect("moving toward"),
    ];

    let selected = core.select_at(&moving, OBSERVATION_CENTER, 1_020_000_000);

    assert_eq!(selected.target_track_id, Some(locked));
    assert_eq!(selected.target_aim_x, Some(280.0));
    assert!(!selected.target_rebuilt);
}

#[test]
fn aim_point_policy_change_rebuilds_prediction_history_without_restarting_for_other_tuning() {
    let mut core = TargetingCore::new(TargetingConfig::default());
    let detection = Detection::new(1, 0, 280.0, 280.0, 80.0, 100.0, 0.9).unwrap();
    core.select_at(&[detection], OBSERVATION_CENTER, 1_000_000_000);
    let stable = core.select_at(
        &[Detection::new(2, 0, 280.0, 280.0, 80.0, 100.0, 0.9).unwrap()],
        OBSERVATION_CENTER,
        1_010_000_000,
    );
    assert!(!stable.target_rebuilt);

    let mut config = TargetingConfig {
        aim_y_ratio: 0.62,
        ..TargetingConfig::default()
    };
    core.set_config(config.clone());
    let rebuilt = core.select_at(
        &[Detection::new(3, 0, 280.0, 280.0, 80.0, 100.0, 0.9).unwrap()],
        OBSERVATION_CENTER,
        1_020_000_000,
    );
    assert!(rebuilt.target_rebuilt);
    assert_eq!(rebuilt.target_aim_y, Some(342.0));

    config.switch_delay_ms = 75.0;
    core.set_config(config);
    let same_target = core.select_at(
        &[Detection::new(4, 0, 280.0, 280.0, 80.0, 100.0, 0.9).unwrap()],
        OBSERVATION_CENTER,
        1_030_000_000,
    );
    assert!(!same_target.target_rebuilt);
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
    assert_eq!(selection.target_aim_y, Some(230.4));
    assert_eq!(selection.target_box_x, Some(280.0));
    assert_eq!(selection.target_box_y, Some(200.0));
    assert_eq!(selection.target_box_width, Some(80.0));
    assert_eq!(selection.target_box_height, Some(100.0));
    let track = core.locked().expect("track");
    assert_eq!(track.center_y, 250.0);
    assert_eq!(track.observed_aim_y, 230.4);
}

#[test]
fn aim_point_motion_from_bbox_shape_does_not_rebuild_identity() {
    let mut core = TargetingCore::new(TargetingConfig {
        aim_y_ratio: 0.0,
        ..TargetingConfig::default()
    });
    let compact = Detection::new(1, 0, 300.0, 300.0, 40.0, 40.0, 0.95).expect("compact");
    let first = core.select_at(&[compact], OBSERVATION_CENTER, 1_000_000_000);
    let track_id = first.target_track_id.expect("initial track");

    // The geometric center is unchanged, but a top-edge aim point moves 30px
    // when height changes. Identity association must remain bbox-centered.
    let tall = Detection::new(2, 0, 300.0, 270.0, 40.0, 100.0, 0.95).expect("tall");
    let second = core.select_at(&[tall], OBSERVATION_CENTER, 1_010_000_000);
    assert_eq!(second.target_track_id, Some(track_id));
    assert!(!second.target_rebuilt);
    assert_eq!(second.target_aim_y, Some(270.0));
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
    assert_eq!(first.lock_reason, Some(LockReason::NearbyAim));
    // Head moved 20 px, still within debounce window.
    let frame2 = vec![Detection::new(1, 0, 320.0, 300.0, 40.0, 80.0, 0.9).expect("d2")];
    let second = core.select(&frame2, OBSERVATION_CENTER);
    assert_eq!(second.target_object_id, Some(1));
    assert_eq!(second.lock_reason, Some(LockReason::MaintainedTarget));
}

#[test]
fn fallback_after_circle_exit_requires_capture_time_confirmation() {
    let mut config = TargetingConfig {
        selection_weights: SelectionWeights {
            distance: 0.99,
            class: 0.01,
            confidence: 0.0,
        },
        switch_min_preference_advantage: 0.01,
        switch_delay_ms: 50.0,
        ..TargetingConfig::default()
    };
    let mut core = TargetingCore::new(config.clone());
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

    // The existing lock stays tracked but can no longer drive control.
    config.target_fov_radius_px = 5.0;
    core.set_config(config);
    let challenger_wins = vec![
        Detection::new(11, 0, 288.0, 285.0, 80.0, 160.0, 0.9).expect("locked moved"),
        Detection::new(12, 1, 280.0, 285.0, 80.0, 160.0, 0.9).expect("stable challenger"),
    ];
    let pending = core.select_at(&challenger_wins, OBSERVATION_CENTER, 1_020_000_000);
    assert_eq!(
        pending.target_object_id, None,
        "outside lock must pause output"
    );

    for timestamp in [1_040_000_000, 1_080_000_000, 1_120_000_000] {
        core.select_at(&challenger_wins, OBSERVATION_CENTER, timestamp);
    }
    let committed = core.select_at(&challenger_wins, OBSERVATION_CENTER, 1_180_000_000);
    assert_eq!(
        committed.target_object_id,
        Some(12),
        "capture-time advantage held beyond 50 ms must switch"
    );
    assert_eq!(committed.lock_reason, Some(LockReason::MaintainedTarget));
}

#[test]
fn nearby_parts_override_preference_and_valid_lock_survives_until_release() {
    for scale in [0.5, 1.0, 2.0] {
        for preferred in [0, 1] {
            let center = (320.0 * scale, 320.0 * scale);
            let mut core = TargetingCore::new(TargetingConfig {
                target_fov_radius_px: 180.0 * scale,
                aim_y_ratio: 0.5,
                class_weights: BTreeMap::from([(preferred, 1.0)]),
                ..Default::default()
            });
            let body = Detection::new(
                1,
                0,
                (300.0 * scale) as f32,
                (300.0 * scale) as f32,
                (40.0 * scale) as f32,
                (160.0 * scale) as f32,
                0.95,
            )
            .unwrap();
            let head = Detection::new(
                2,
                1,
                (310.0 * scale) as f32,
                (310.0 * scale) as f32,
                (20.0 * scale) as f32,
                (20.0 * scale) as f32,
                0.95,
            )
            .unwrap();
            // The head is also inside the body rectangle. Selection uses aim
            // point distance, not rectangle containment or the class label.
            core.select_at(&[body.clone(), head.clone()], center, 1_000_000_000);
            let acquired = core.select_at(&[head.clone(), body.clone()], center, 1_020_000_000);
            assert_eq!(acquired.target_class_id, Some(1));
            assert_eq!(acquired.lock_reason, Some(LockReason::NearbyAim));
            let held = core.select_at(&[body.clone(), head.clone()], center, 1_080_000_000);
            assert_eq!(held.target_track_id, acquired.target_track_id);
            assert_eq!(held.lock_reason, Some(LockReason::MaintainedTarget));

            core.release_lock();
            let at_body = core.select_at(
                &[head.clone(), body.clone()],
                (center.0, 380.0 * scale),
                1_100_000_000,
            );
            assert_eq!(at_body.target_class_id, Some(0));
            assert_eq!(at_body.lock_reason, Some(LockReason::NearbyAim));

            core.release_lock();
            let far = core.select_at(&[body, head], (center.0, 440.0 * scale), 1_120_000_000);
            assert_eq!(far.target_class_id, Some(preferred));
        }
    }
}

#[test]
fn a_nested_other_class_cannot_inherit_the_selected_part_identity() {
    let mut core = TargetingCore::new(TargetingConfig::default());
    let body = Detection::new(1, 0, 280.0, 280.0, 80.0, 160.0, 0.95).unwrap();
    let locked = core
        .select_at(&[body], OBSERVATION_CENTER, 1_000_000_000)
        .target_track_id
        .unwrap();
    // Size and center gates alone permit this pair, but box overlap is low.
    let head = Detection::new(2, 1, 290.0, 280.0, 60.0, 80.0, 0.95).unwrap();
    let missing = core.select_at(&[head.clone()], OBSERVATION_CENTER, 1_020_000_000);
    assert_eq!(missing.target_track_id, None);
    assert_eq!(core.locked().unwrap().id, locked);
    core.select_at(&[head.clone()], OBSERVATION_CENTER, 1_040_000_000);
    let fallback = core.select_at(&[head], OBSERVATION_CENTER, 1_100_000_000);
    assert_eq!(fallback.target_class_id, Some(1));
    assert_ne!(fallback.target_track_id, Some(locked));
}

#[test]
fn proven_head_body_handoff_keeps_subject_identity_and_marks_part_change() {
    let mut core = TargetingCore::new(TargetingConfig {
        class_roles: BTreeMap::from([(0, TargetPartRole::Head), (1, TargetPartRole::Body)]),
        switch_delay_ms: 0.0,
        switch_min_continuity_score: 0.0,
        ..TargetingConfig::default()
    });
    let body = Detection::new(1, 1, 280.0, 280.0, 80.0, 160.0, 0.95).unwrap();
    let head = Detection::new(2, 0, 295.0, 280.0, 50.0, 50.0, 0.95).unwrap();
    core.select_at(
        &[body.clone(), head.clone()],
        OBSERVATION_CENTER,
        1_000_000_000,
    );
    let body_lock = core.select_at(&[head.clone(), body], OBSERVATION_CENTER, 1_010_000_000);
    let body_track = body_lock.target_track_id.expect("body part track");
    let subject = body_lock.target_subject_id.expect("physical subject");
    assert_eq!(body_lock.target_class_id, Some(1));

    let head_handoff = core.select_at(&[head.clone()], OBSERVATION_CENTER, 1_020_000_000);
    assert_ne!(head_handoff.target_track_id, Some(body_track));
    assert_eq!(head_handoff.target_subject_id, Some(subject));
    assert_eq!(head_handoff.target_class_id, Some(0));
    assert!(head_handoff.target_part_changed);

    let stable_head = core.select_at(&[head], OBSERVATION_CENTER, 1_030_000_000);
    assert_eq!(stable_head.target_subject_id, Some(subject));
    assert!(!stable_head.target_part_changed);
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
fn same_class_identity_confidence_is_independent_of_class_association_weight() {
    let mut confidences = Vec::new();
    for class_weight in [0.0, 0.35, 2.0] {
        let mut core = TargetingCore::new(TargetingConfig {
            tracker_class_cost_weight: class_weight,
            ..TargetingConfig::default()
        });
        let first = Detection::new(1, 0, 300.0, 300.0, 40.0, 80.0, 0.95).unwrap();
        let locked = core
            .select_at(&[first], OBSERVATION_CENTER, 1_000_000_000)
            .target_track_id
            .expect("initial identity");
        let moved = Detection::new(2, 0, 318.0, 300.0, 40.0, 80.0, 0.95).unwrap();
        let selected = core.select_at(&[moved], OBSERVATION_CENTER, 1_010_000_000);
        assert_eq!(selected.target_track_id, Some(locked));
        assert!(!selected.target_rebuilt);
        let confidence = selected
            .target_identity_confidence
            .expect("current observed identity confidence");
        assert!(confidence > 0.0 && confidence < 1.0);
        confidences.push(confidence);
    }

    for confidence in &confidences[1..] {
        assert!(
            (confidence - confidences[0]).abs() < 1e-12,
            "class cost must not inflate same-class geometry confidence: {confidences:?}"
        );
    }
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
    let mut core = TargetingCore::new(TargetingConfig::default());
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
fn confirmed_challenger_takes_over_before_missing_lock_grace_expires() {
    let mut core = TargetingCore::new(TargetingConfig {
        track_max_lost_age_ms: 120.0,
        switch_delay_ms: 50.0,
        ..TargetingConfig::default()
    });
    let both_visible = [
        Detection::new(1, 0, 280.0, 270.0, 80.0, 100.0, 0.95).expect("locked"),
        Detection::new(2, 0, 380.0, 270.0, 80.0, 100.0, 0.95).expect("challenger"),
    ];
    core.select_at(&both_visible, OBSERVATION_CENTER, 1_000_000_000);
    let locked = core
        .select_at(&both_visible, OBSERVATION_CENTER, 1_010_000_000)
        .target_track_id
        .expect("locked track");
    let challenger =
        Detection::new(12, 0, 380.0, 270.0, 80.0, 100.0, 0.95).expect("challenger only");

    let waiting = core.select_at(
        std::slice::from_ref(&challenger),
        OBSERVATION_CENTER,
        1_020_000_000,
    );
    assert_eq!(waiting.target_track_id, None);
    let switched = core.select_at(&[challenger], OBSERVATION_CENTER, 1_071_000_000);

    assert!(switched.target_track_id.is_some());
    assert_ne!(switched.target_track_id, Some(locked));
    assert_eq!(switched.target_object_id, Some(12));
}

#[test]
fn locked_target_loss_grace_uses_capture_time_instead_of_inference_frame_count() {
    let mut core = TargetingCore::new(TargetingConfig {
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
        .select(&[first], (120.0, 140.0))
        .target_track_id
        .expect("first identity");
    let jumped = Detection::new(2, 0, 221.0, 100.0, 40.0, 80.0, 0.9).expect("jumped");
    let next = core.select(&[jumped], (241.0, 140.0));
    assert_ne!(next.target_track_id, Some(first_track));
    assert_eq!(core.locked().expect("new track").identity_confidence, 1.0);
}

#[test]
fn filtered_classes_age_out_the_previous_track() {
    let mut core = TargetingCore::new(TargetingConfig {
        allowed_class_ids: Some(BTreeSet::from([0, 1])),
        ..TargetingConfig::default()
    });
    let target = Detection::new(0, 0, 280.0, 300.0, 40.0, 80.0, 0.9).expect("target");
    let first = core.select(std::slice::from_ref(&target), OBSERVATION_CENTER);
    let first_track = first.target_track_id.expect("first track");
    let irrelevant = Detection::new(0, 2, 280.0, 300.0, 40.0, 80.0, 0.9).expect("irrelevant");
    for _ in 0..6 {
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
fn crowded_frame_can_choose_a_late_better_class_without_input_order_bias() {
    let mut candidates = (0..16)
        .map(|index| {
            Detection::new(index + 1, 0, 150.0, 280.0, 40.0, 80.0, 0.9).expect("distractor")
        })
        .collect::<Vec<_>>();
    candidates.push(Detection::new(17, 1, 300.0, 280.0, 40.0, 80.0, 0.9).expect("better"));

    for frame in [candidates.clone(), {
        let mut reversed = candidates.clone();
        reversed.reverse();
        reversed
    }] {
        let mut core = TargetingCore::new(TargetingConfig::default());
        core.select_at(&frame, OBSERVATION_CENTER, 1_000_000_000);
        let selection = core.select_at(&frame, OBSERVATION_CENTER, 1_010_000_000);
        assert_eq!(selection.target_object_id, Some(17));
        assert_eq!(selection.inside_fov, 17);
        assert_eq!(selection.admitted_to_tracking, 16);
        assert_eq!(selection.dropped_by_budget, 1);
    }
}

#[test]
fn crowded_frame_keeps_the_existing_lock_in_the_association_budget() {
    let mut core = TargetingCore::new(TargetingConfig::default());
    let incumbent = Detection::new(100, 2, 300.0, 280.0, 40.0, 80.0, 0.9).expect("incumbent");
    let locked = core.select_at(
        std::slice::from_ref(&incumbent),
        OBSERVATION_CENTER,
        1_000_000_000,
    );
    let locked_id = locked.target_track_id.expect("initial lock");
    let mut crowded = (0..16)
        .map(|index| {
            Detection::new(index + 1, 0, 280.0, 280.0, 40.0, 80.0, 0.9).expect("distractor")
        })
        .collect::<Vec<_>>();
    crowded.push(incumbent);

    let selection = core.select_at(&crowded, OBSERVATION_CENTER, 1_010_000_000);
    assert_eq!(selection.inside_fov, 17);
    assert_eq!(selection.dropped_by_budget, 1);
    assert_eq!(selection.target_object_id, Some(100));
    assert_eq!(selection.target_track_id, Some(locked_id));
}

#[test]
fn lost_track_cannot_produce_a_target_object_id() {
    let mut core = TargetingCore::new(TargetingConfig::default());
    let head = Detection::new(1, 0, 300.0, 280.0, 40.0, 80.0, 0.9).expect("head");
    core.select(&[head], OBSERVATION_CENTER);
    // Timestamp-free replay keeps a short fixed internal grace period. Once
    // those frames are consumed, the next admissible frame starts cleanly.
    for _ in 0..5 {
        core.select(&[], OBSERVATION_CENTER);
    }
    let body = Detection::new(2, 1, 320.0, 360.0, 30.0, 60.0, 0.9).expect("body");
    let next = core.select(&[body], OBSERVATION_CENTER);
    assert_eq!(next.target_object_id, Some(2));
    assert_eq!(next.target_class_id, Some(1));
}
