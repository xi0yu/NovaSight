from __future__ import annotations

import math

import pytest

from novasight.contracts import Detection, FrameContext
from novasight.runtime.candidates import BasicCandidateFilter
from novasight.runtime.kalman import KalmanConfig
from novasight.runtime.target_selector import RuntimeTargetSelector
from novasight.runtime.tracker import RuntimeTracker, TrackerConfig, _linear_sum_assignment


def _context(
    frame_id: int,
    capture_ts_ns: int,
    detections: list[Detection],
) -> FrameContext:
    return FrameContext(
        frame_id=frame_id,
        capture_ts_ns=capture_ts_ns,
        width=640,
        height=640,
        detections=detections,
    )


def _observations(
    context: FrameContext,
    *,
    aim_y_ratio: float = 0.22,
    allowed_class_ids: set[int] | None = None,
    min_confidence: float = 0.25,
):
    return BasicCandidateFilter().apply(
        context,
        allowed_class_ids=allowed_class_ids,
        min_confidence=min_confidence,
        aim_y_ratio=aim_y_ratio,
    )


def _tracker(**overrides) -> RuntimeTracker:
    config = TrackerConfig(
        max_match_distance=float(overrides.get("max_match_distance", 1.5)),
        position_cost_weight=float(overrides.get("position_cost_weight", 0.75)),
        iou_cost_weight=float(overrides.get("iou_cost_weight", 0.25)),
        max_missed_frames=int(overrides.get("max_missed_frames", 2)),
        kalman=KalmanConfig(
            acceleration_noise=2.0,
            measurement_noise_x=4.0,
            measurement_noise_y=4.0,
        ),
    )
    return RuntimeTracker(config)


def test_basic_candidate_filter_only_applies_class_confidence_and_bbox_validity() -> None:
    context = _context(
        1,
        1_000_000_000,
        [
            Detection(1, 0.90, x=100, y=200, w=40, h=100),
            Detection(2, 0.95, x=300, y=200, w=40, h=100),
            Detection(1, 0.20, x=500, y=200, w=40, h=100),
        ],
    )

    result = _observations(
        context,
        aim_y_ratio=0.22,
        allowed_class_ids={1},
        min_confidence=0.25,
    )

    assert len(result.observations) == 1
    observation = result.observations[0]
    assert observation.detection_index == 0
    assert observation.aim_x == pytest.approx(120.0)
    assert observation.aim_y == pytest.approx(222.0)
    assert {item["reason"] for item in result.rejected} == {
        "class_filter",
        "confidence_filter",
    }


def test_candidate_aim_uses_default_ratio_with_per_class_override() -> None:
    context = _context(
        1,
        1_000_000_000,
        [
            Detection(0, 0.9, x=100, y=200, w=40, h=100),
            Detection(1, 0.9, x=200, y=200, w=40, h=100),
        ],
    )

    result = BasicCandidateFilter().apply(
        context,
        allowed_class_ids=None,
        min_confidence=0.25,
        aim_y_ratio=0.22,
        class_aim_y_ratios={1: 0.50},
    )

    assert result.observations[0].aim_y == pytest.approx(222.0)
    assert result.observations[1].aim_y == pytest.approx(250.0)


def test_single_high_confidence_track_is_confirmed_immediately() -> None:
    tracker = _tracker()
    context = _context(
        1,
        1_000_000_000,
        [Detection(0, 0.9, x=100, y=200, w=40, h=100)],
    )
    observations = _observations(context)

    result = tracker.update(
        observations.observations,
        context.capture_ts_ns or 0,
        frame_id=context.frame_id,
    )

    assert result.state == "TRACKING"
    assert result.created_track_ids == [1]
    assert len(result.active_tracks) == 1
    assert result.active_tracks[0].track_id == 1
    assert result.active_tracks[0].observed_aim_px == pytest.approx((120.0, 222.0))
    assert result.active_tracks[0].velocity_valid is False
    assert result.debug["tracks"][0]["status"] == "CONFIRMED"


def test_multiple_new_tracks_require_second_hit_before_control_output() -> None:
    tracker = _tracker()
    first = _context(
        1,
        1_000_000_000,
        [
            Detection(0, 0.9, x=100, y=200, w=40, h=100),
            Detection(0, 0.9, x=300, y=200, w=40, h=100),
        ],
    )
    second = _context(
        2,
        1_050_000_000,
        [
            Detection(0, 0.9, x=104, y=200, w=40, h=100),
            Detection(0, 0.9, x=304, y=200, w=40, h=100),
        ],
    )

    tentative = tracker.update(
        _observations(first).observations,
        first.capture_ts_ns or 0,
        frame_id=first.frame_id,
    )
    confirmed = tracker.update(
        _observations(second).observations,
        second.capture_ts_ns or 0,
        frame_id=second.frame_id,
    )

    assert tentative.active_tracks == []
    assert tentative.state == "ACQUIRING"
    assert [item["status"] for item in tentative.debug["tracks"]] == ["TENTATIVE", "TENTATIVE"]
    assert [track.track_id for track in confirmed.active_tracks] == [1, 2]
    assert [item["status"] for item in confirmed.debug["tracks"]] == ["CONFIRMED", "CONFIRMED"]


def test_capture_timestamp_delta_updates_same_track_velocity() -> None:
    tracker = _tracker()
    first = _context(
        1,
        1_000_000_000,
        [Detection(0, 0.9, x=100, y=200, w=40, h=100)],
    )
    second = _context(
        2,
        1_100_000_000,
        [Detection(0, 0.9, x=110, y=200, w=40, h=100)],
    )

    tracker.update(_observations(first).observations, first.capture_ts_ns or 0, frame_id=1)
    result = tracker.update(_observations(second).observations, second.capture_ts_ns or 0, frame_id=2)

    track = result.active_tracks[0]
    assert track.track_id == 1
    assert track.velocity_valid is True
    assert track.velocity_px_s[0] > 0.0
    assert result.debug["tracks"][0]["last_capture_ts_ns"] == 1_100_000_000


def test_lost_track_is_not_output_and_restores_same_track_id() -> None:
    tracker = _tracker(max_missed_frames=2)
    first = _context(
        1,
        1_000_000_000,
        [Detection(0, 0.9, x=100, y=200, w=40, h=100)],
    )
    missing = _context(2, 1_050_000_000, [])
    restored = _context(
        3,
        1_100_000_000,
        [Detection(0, 0.9, x=108, y=200, w=40, h=100)],
    )

    tracker.update(_observations(first).observations, first.capture_ts_ns or 0, frame_id=1)
    lost = tracker.update([], missing.capture_ts_ns or 0, frame_id=2)
    recovered = tracker.update(
        _observations(restored).observations,
        restored.capture_ts_ns or 0,
        frame_id=3,
    )

    assert lost.active_tracks == []
    assert lost.lost_track_count == 1
    assert lost.debug["tracks"][0]["status"] == "LOST"
    assert recovered.restored_track_ids == [1]
    assert recovered.active_tracks[0].track_id == 1
    assert recovered.active_tracks[0].track_rebuilt is True
    assert recovered.debug["tracks"][0]["status"] == "CONFIRMED"


def test_track_is_removed_after_max_missed_frames() -> None:
    tracker = _tracker(max_missed_frames=2)
    first = _context(
        1,
        1_000_000_000,
        [Detection(0, 0.9, x=100, y=200, w=40, h=100)],
    )
    tracker.update(_observations(first).observations, first.capture_ts_ns or 0, frame_id=1)

    tracker.update([], 1_100_000_000, frame_id=2)
    tracker.update([], 1_200_000_000, frame_id=3)
    removed = tracker.update([], 1_300_000_000, frame_id=4)

    assert removed.removed_track_ids == [1]
    assert removed.active_tracks == []
    assert removed.lost_track_count == 0
    assert removed.state == "IDLE"


def test_normalized_distance_gate_prevents_cross_screen_reassociation() -> None:
    tracker = _tracker(max_match_distance=1.0)
    first = _context(
        1,
        1_000_000_000,
        [Detection(0, 0.9, x=50, y=200, w=40, h=80)],
    )
    jump = _context(
        2,
        1_100_000_000,
        [Detection(0, 0.9, x=500, y=200, w=40, h=80)],
    )

    tracker.update(_observations(first).observations, first.capture_ts_ns or 0, frame_id=1)
    result = tracker.update(_observations(jump).observations, jump.capture_ts_ns or 0, frame_id=2)

    assert result.created_track_ids == [2]
    assert [track.track_id for track in result.active_tracks] == [2]
    assert result.lost_track_count == 1


def test_size_gate_prevents_implausible_bbox_reassociation() -> None:
    tracker = _tracker()
    first = _context(
        1,
        1_000_000_000,
        [Detection(0, 0.9, x=100, y=200, w=40, h=100)],
    )
    size_jump = _context(
        2,
        1_050_000_000,
        [Detection(0, 0.9, x=40, y=134, w=160, h=400)],
    )

    tracker.update(_observations(first).observations, first.capture_ts_ns or 0, frame_id=1)
    result = tracker.update(
        _observations(size_jump).observations,
        size_jump.capture_ts_ns or 0,
        frame_id=2,
    )

    assert result.created_track_ids == [2]
    assert [track.track_id for track in result.active_tracks] == [2]
    assert result.lost_track_count == 1


def test_mahalanobis_gate_rejects_statistically_impossible_match() -> None:
    tracker = _tracker(max_match_distance=100.0)
    first = _context(
        1,
        1_000_000_000,
        [Detection(0, 0.9, x=100, y=200, w=40, h=100)],
    )
    jump = _context(
        2,
        1_050_000_000,
        [Detection(0, 0.9, x=300, y=200, w=40, h=100)],
    )

    tracker.update(_observations(first).observations, first.capture_ts_ns or 0, frame_id=1)
    result = tracker.update(_observations(jump).observations, jump.capture_ts_ns or 0, frame_id=2)

    assert result.created_track_ids == [2]
    assert [track.track_id for track in result.active_tracks] == [2]
    assert result.lost_track_count == 1


def test_invalid_kalman_prediction_falls_back_to_last_observation_for_association() -> None:
    tracker = RuntimeTracker(
        TrackerConfig(
            max_match_distance=1.5,
            max_association_dt_ms=150.0,
            kalman=KalmanConfig(
                acceleration_noise=2.0,
                measurement_noise_x=4.0,
                measurement_noise_y=4.0,
                prediction_decay_tau_ms=10.0,
                min_prediction_confidence=0.90,
            ),
        )
    )
    first = _context(
        1,
        1_000_000_000,
        [Detection(0, 0.9, x=100, y=200, w=40, h=100)],
    )
    same_target_after_gap = _context(
        2,
        1_100_000_000,
        [Detection(0, 0.9, x=100, y=200, w=40, h=100)],
    )

    tracker.update(_observations(first).observations, first.capture_ts_ns or 0, frame_id=1)
    track_before_gap = tracker._tracks[1]
    assert track_before_gap.estimator is not None
    track_before_gap.estimator.x[2] = 1_000.0
    result = tracker.update(
        _observations(same_target_after_gap).observations,
        same_target_after_gap.capture_ts_ns or 0,
        frame_id=2,
    )

    assert [track.track_id for track in result.active_tracks] == [1]
    assert result.debug["assignments"][0]["prediction_used"] is False
    recovered = result.active_tracks[0]
    assert recovered.filtered_aim_px == pytest.approx(recovered.observed_aim_px)
    recovered_record = tracker._tracks[1]
    assert recovered_record.estimator is not None
    assert recovered_record.estimator.x[0] == pytest.approx(recovered.observed_aim_px[0])


def test_hungarian_assignment_finds_global_optimum_where_greedy_fails() -> None:
    matrix = [
        [1.0, 2.0],
        [1.1, 100.0],
    ]

    assignment = _linear_sum_assignment(matrix)

    assert assignment == [(0, 1), (1, 0)]
    assert sum(matrix[row][column] for row, column in assignment) == pytest.approx(3.1)


def test_tracker_caps_association_work_and_reports_timing_and_quality() -> None:
    tracker = _tracker()
    detections = [
        Detection(0, 0.99 - index * 0.01, x=index * 20, y=200, w=16, h=80)
        for index in range(20)
    ]
    context = _context(1, 1_000_000_000, detections)

    tentative = tracker.update(
        _observations(context).observations,
        context.capture_ts_ns or 0,
        frame_id=context.frame_id,
    )
    confirmed_context = _context(2, 1_050_000_000, detections)
    result = tracker.update(
        _observations(confirmed_context).observations,
        confirmed_context.capture_ts_ns or 0,
        frame_id=confirmed_context.frame_id,
    )

    assert tentative.active_tracks == []
    assert tentative.debug["tentative_tracks"] == 16
    assert len(result.active_tracks) == 16
    assert result.debug["input_candidates"] == 20
    assert result.debug["association_candidates"] == 16
    assert result.debug["association_candidates_dropped"] == 4
    assert result.debug["max_active_tracks"] == 16
    assert result.debug["max_detections_for_association"] == 16
    first_track = result.debug["tracks"][0]
    sigma = first_track["estimate"]["position_sigma_px"]
    expected_quality = 0.99 / (1.0 + sigma / math.sqrt(16.0 * 80.0))
    assert first_track["track_quality"] == pytest.approx(expected_quality)
    assert result.active_tracks[0].quality_score == pytest.approx(expected_quality)
    for key in (
        "tracker_predict_us",
        "association_matrix_us",
        "hungarian_us",
        "tracker_update_us",
        "tracker_total_us",
    ):
        assert result.debug["timing"][key] >= 0.0


def test_tracker_keeps_configured_kalman_confidence_gate_for_control_estimate() -> None:
    tracker = RuntimeTracker(
        TrackerConfig(
            max_match_distance=1.5,
            kalman=KalmanConfig(
                acceleration_noise=2.0,
                measurement_noise_x=4.0,
                measurement_noise_y=4.0,
                min_identity_confidence=0.99,
            ),
        )
    )
    first = _context(
        1,
        1_000_000_000,
        [Detection(0, 0.9, x=100, y=200, w=40, h=100)],
    )
    second = _context(
        2,
        1_050_000_000,
        [Detection(0, 0.9, x=104, y=200, w=40, h=100)],
    )

    tracker.update(_observations(first).observations, first.capture_ts_ns or 0, frame_id=1)
    result = tracker.update(
        _observations(second).observations,
        second.capture_ts_ns or 0,
        frame_id=2,
    )

    assert result.active_tracks[0].track_id == 1
    assert result.debug["association_algorithm"] == "hungarian"
    assert result.debug["tracks"][0]["identity_confidence"] < 0.99
    assert result.debug["tracks"][0]["estimate"]["valid"] is False


def test_target_selector_never_controls_lost_track() -> None:
    selector = RuntimeTargetSelector()
    first = _context(
        1,
        1_000_000_000,
        [Detection(0, 0.9, x=300, y=240, w=40, h=100)],
    )
    missing = _context(2, 1_050_000_000, [])
    restored = _context(
        3,
        1_100_000_000,
        [Detection(0, 0.9, x=304, y=240, w=40, h=100)],
    )

    selected = selector.select(
        first,
        min_confidence=0.25,
        fov_ratio=1.0,
        aim_ratio=0.22,
        tracker_max_missed_frames=2,
        target_switch_delay_ms=0,
    )
    lost = selector.select(
        missing,
        min_confidence=0.25,
        fov_ratio=1.0,
        aim_ratio=0.22,
        tracker_max_missed_frames=2,
        target_switch_delay_ms=0,
    )
    lost_track_status = selector.last_debug["tracker"]["tracks"][0]["status"]
    recovered = selector.select(
        restored,
        min_confidence=0.25,
        fov_ratio=1.0,
        aim_ratio=0.22,
        tracker_max_missed_frames=2,
        target_switch_delay_ms=0,
    )

    assert selected.target is not None
    assert selected.target.track_id == 1
    assert lost.target is None
    assert lost_track_status == "LOST"
    assert recovered.target is not None
    assert recovered.target.track_id == 1
    assert selector.last_debug["tracker"]["restored_track_ids"] == [1]


def test_initial_target_is_committed_without_switch_delay() -> None:
    selector = RuntimeTargetSelector()
    context = _context(
        1,
        1_000_000_000,
        [Detection(0, 0.9, x=300, y=240, w=40, h=100)],
    )

    selection = selector.select(
        context,
        min_confidence=0.25,
        fov_ratio=1.0,
        aim_ratio=0.22,
        target_switch_delay_ms=500,
    )

    assert selection.target is not None
    assert selection.state == "acquire"
    assert selection.reason == "initial target committed"


def test_target_selector_uses_explicit_control_center_inside_shifted_roi() -> None:
    selector = RuntimeTargetSelector()
    context = _context(
        1,
        1_000_000_000,
        [
            Detection(0, 0.9, x=300, y=240, w=40, h=100),
            Detection(0, 0.9, x=340, y=240, w=40, h=100),
        ],
    )

    selection = selector.select(
        context,
        min_confidence=0.25,
        fov_ratio=1.0,
        aim_ratio=0.22,
        control_center_x_px=360.0,
        control_center_y_px=320.0,
        target_switch_delay_ms=0,
    )
    confirmed_context = _context(2, 1_050_000_000, list(context.detections))
    selection = selector.select(
        confirmed_context,
        min_confidence=0.25,
        fov_ratio=1.0,
        aim_ratio=0.22,
        control_center_x_px=360.0,
        control_center_y_px=320.0,
        target_switch_delay_ms=0,
    )

    assert selection.target is not None
    assert selection.target.cx == pytest.approx(360.0)
    assert selector.last_debug["control_center_roi_px"] == {"x": 360.0, "y": 320.0}


def test_same_class_fallback_accepts_small_nearest_candidate_over_larger_bbox() -> None:
    selector = RuntimeTargetSelector()
    detections = [
        # 128 px² was below the removed 256 px² hard area gate.
        Detection(0, 0.90, x=316, y=304, w=8, h=16),
        Detection(0, 0.90, x=500, y=180, w=100, h=200),
    ]
    first = _context(1, 1_000_000_000, detections)
    second = _context(2, 1_050_000_000, detections)

    selector.select(
        first,
        min_confidence=0.25,
        fov_ratio=1.0,
        aim_ratio=0.22,
        class_priority=[1, 0],
        target_switch_delay_ms=0,
    )
    selected = selector.select(
        second,
        min_confidence=0.25,
        fov_ratio=1.0,
        aim_ratio=0.22,
        class_priority=[1, 0],
        target_switch_delay_ms=0,
    )

    assert selected.target is not None
    assert selected.target.box == detections[0].box
    candidates = selector.last_debug["tracked_filter"]["candidates"]
    near = next(item for item in candidates if item["x"] == 316.0)
    far = next(item for item in candidates if item["x"] == 500.0)
    assert near["distance_score"] > far["distance_score"]
    assert near["selection_score"] > far["selection_score"]


def test_lost_preferred_head_falls_back_to_nearest_body() -> None:
    selector = RuntimeTargetSelector()
    with_head = [
        Detection(1, 0.90, x=300, y=240, w=40, h=100),
        Detection(0, 0.72, x=310, y=270, w=20, h=50),
        Detection(0, 0.99, x=470, y=170, w=110, h=220),
    ]
    bodies_only = with_head[1:]
    common = {
        "min_confidence": 0.25,
        "fov_ratio": 1.0,
        "aim_ratio": 0.22,
        "class_priority": [1, 0],
        "lost_grace_frames": 0,
        "tracker_max_missed_frames": 0,
        "target_switch_delay_ms": 0,
    }

    selector.select(_context(1, 1_000_000_000, with_head), **common)
    head_selection = selector.select(_context(2, 1_050_000_000, with_head), **common)
    fallback = selector.select(_context(3, 1_100_000_000, bodies_only), **common)

    assert head_selection.target is not None
    assert head_selection.target.cls == 1
    assert fallback.target is not None
    assert fallback.target.cls == 0
    assert fallback.target.box == bodies_only[0].box


def test_switch_debounce_keeps_valid_locked_target_until_challenger_commits() -> None:
    selector = RuntimeTargetSelector()
    first = _context(
        1,
        1_000_000_000,
        [Detection(0, 0.90, x=280, y=240, w=40, h=100)],
    )
    challenger = _context(
        2,
        1_020_000_000,
        [
            Detection(0, 0.90, x=280, y=240, w=40, h=100),
            Detection(1, 0.95, x=300, y=240, w=40, h=100),
        ],
    )
    confirmed_challenger = _context(
        3,
        1_140_000_000,
        [
            Detection(0, 0.90, x=280, y=240, w=40, h=100),
            Detection(1, 0.95, x=300, y=240, w=40, h=100),
        ],
    )
    committed_challenger = _context(
        4,
        1_260_000_000,
        [
            Detection(0, 0.90, x=280, y=240, w=40, h=100),
            Detection(1, 0.95, x=300, y=240, w=40, h=100),
        ],
    )

    initial = selector.select(
        first,
        min_confidence=0.25,
        fov_ratio=1.0,
        aim_ratio=0.22,
        class_priority=[1, 0],
        target_switch_delay_ms=100,
    )
    pending = selector.select(
        challenger,
        min_confidence=0.25,
        fov_ratio=1.0,
        aim_ratio=0.22,
        class_priority=[1, 0],
        target_switch_min_preference_advantage=0.0,
        target_switch_min_continuity_score=0.0,
        target_switch_delay_ms=100,
    )
    confirmed = selector.select(
        confirmed_challenger,
        min_confidence=0.25,
        fov_ratio=1.0,
        aim_ratio=0.22,
        class_priority=[1, 0],
        target_switch_min_preference_advantage=0.0,
        target_switch_min_continuity_score=0.0,
        target_switch_delay_ms=100,
    )
    committed = selector.select(
        committed_challenger,
        min_confidence=0.25,
        fov_ratio=1.0,
        aim_ratio=0.22,
        class_priority=[1, 0],
        target_switch_min_preference_advantage=0.0,
        target_switch_min_continuity_score=0.0,
        target_switch_delay_ms=100,
    )

    assert initial.target is not None
    assert pending.target is not None
    assert pending.target.track_id == initial.target.track_id
    assert pending.state == "locked"
    assert pending.locked is True
    assert confirmed.state == "switch_hold"
    assert confirmed.locked is True
    assert committed.target is not None
    assert committed.target.track_id != initial.target.track_id
    assert committed.state == "switch_committed"
