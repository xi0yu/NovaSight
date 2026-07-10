from __future__ import annotations

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


def test_new_track_is_active_immediately_and_velocity_starts_invalid() -> None:
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
    assert result.debug["tracks"][0]["status"] == "ACTIVE"


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
    missing = _context(2, 1_100_000_000, [])
    restored = _context(
        3,
        1_200_000_000,
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
    assert recovered.debug["tracks"][0]["status"] == "ACTIVE"


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


def test_hungarian_assignment_finds_global_optimum_where_greedy_fails() -> None:
    matrix = [
        [1.0, 2.0],
        [1.1, 100.0],
    ]

    assignment = _linear_sum_assignment(matrix)

    assert assignment == [(0, 1), (1, 0)]
    assert sum(matrix[row][column] for row, column in assignment) == pytest.approx(3.1)


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
    missing = _context(2, 1_100_000_000, [])
    restored = _context(
        3,
        1_200_000_000,
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
