from __future__ import annotations

from dataclasses import dataclass
from statistics import median
import time
from typing import Iterable

from novasight.contracts import FrameContext, Track
from novasight.runtime.candidates import (
    AssociationCandidateFilter,
    BasicCandidateFilter,
    CandidateQuality,
    QualityScoreConfig,
    QualityScorer,
    ScoredTrack,
    TrackScoreResult,
    parse_allowed_class_ids,
    scored_track_debug,
)
from novasight.runtime.kalman import KalmanConfig
from novasight.runtime.tracker import RuntimeTracker, TrackerConfig, TrackerUpdate


def _context_time_ns(context: FrameContext) -> int:
    capture_ts_ns = context.capture_ts_ns
    if isinstance(capture_ts_ns, int) and capture_ts_ns > 0:
        return capture_ts_ns
    return time.monotonic_ns()


@dataclass(frozen=True)
class TargetSelection:
    target: Track | None
    state: str
    reason: str
    candidates: int = 0
    inside_fov: int = 0
    locked: bool = False
    lost_count: int = 0
    priority_rank: int | None = None
    distance_px: float | None = None
    quality_score: float | None = None
    class_score: float | None = None
    distance_score: float | None = None
    selection_score: float | None = None


@dataclass
class _LockedTarget:
    key: str


@dataclass
class _PendingSwitch:
    key: str
    started_ts_ns: int
    elapsed_ms: float
    advantage: float
    continuity_score: float
    reason: str


class RuntimeTargetSelector:
    def __init__(self) -> None:
        self._locked: _LockedTarget | None = None
        self._pending_switch: _PendingSwitch | None = None
        self._lost_count = 0
        self.last_debug: dict = {}
        self.tracker = RuntimeTracker()

    def reset(self) -> None:
        self._locked = None
        self._pending_switch = None
        self._lost_count = 0
        self.last_debug = {}
        self.tracker.reset()

    def select(
        self,
        context: FrameContext,
        *,
        min_confidence: float,
        fov_ratio: float,
        aim_ratio: float = 0.5,
        class_aim_y_ratios: dict[int, float] | None = None,
        control_center_x_px: float | None = None,
        control_center_y_px: float | None = None,
        class_filter: str = "all",
        class_priority: Iterable[int] = (),
        sticky_bias: float = 0.25,
        lock_enabled: bool = True,
        lost_grace_frames: int = 5,
        ratio_max_aspect: float = 6.0,
        quality_confidence_weight: float = 0.7,
        quality_area_weight: float = 0.3,
        selection_class_weight: float = 0.40,
        selection_quality_weight: float = 0.05,
        selection_distance_weight: float = 0.55,
        tracker_max_match_distance: float = 1.5,
        tracker_position_cost_weight: float = 0.75,
        tracker_iou_cost_weight: float = 0.25,
        tracker_max_missed_frames: int = 2,
        kalman_acceleration_noise: float = 1200.0,
        kalman_measurement_noise_x: float = 16.0,
        kalman_measurement_noise_y: float = 16.0,
        kalman_max_predict_missing_ms: float = 80.0,
        kalman_max_predict_steps: int = 5,
        kalman_max_predict_dt_ms: float = 35.0,
        kalman_max_position_sigma_px: float = 45.0,
        kalman_max_covariance_trace: float = 5000.0,
        kalman_nis_threshold: float = 9.21,
        kalman_nis_hard_reject: float = 16.0,
        kalman_min_identity_confidence: float = 0.70,
        kalman_min_prediction_confidence: float = 0.35,
        kalman_prediction_decay_tau_ms: float = 45.0,
        target_switch_min_preference_advantage: float = 0.08,
        target_switch_min_continuity_score: float = 0.70,
        target_switch_delay_ms: float = 50.0,
    ) -> TargetSelection:
        raw_candidates = list(context.detections)
        if context.width <= 0 or context.height <= 0:
            self._lost_count += 1 if self._locked is not None else 0
            tracker_update = self.tracker.update(
                [],
                int(context.capture_ts_ns or 0),
                frame_id=context.frame_id,
            )
            self.last_debug = {
                "raw_candidates": len(raw_candidates),
                "filtered_candidates": 0,
                "inside_fov": 0,
                "reason": "invalid frame geometry",
                "tracker": tracker_update.debug,
            }
            return self._lost_or_clear(lost_grace_frames, "invalid frame geometry", 0, 0)

        self.tracker.config = TrackerConfig(
            max_match_distance=max(0.0, float(tracker_max_match_distance)),
            position_cost_weight=max(0.0, float(tracker_position_cost_weight)),
            iou_cost_weight=max(0.0, float(tracker_iou_cost_weight)),
            max_missed_frames=max(0, int(tracker_max_missed_frames)),
            kalman=KalmanConfig(
                enabled=True,
                acceleration_noise=max(1e-6, float(kalman_acceleration_noise)),
                measurement_noise_x=max(1e-6, float(kalman_measurement_noise_x)),
                measurement_noise_y=max(1e-6, float(kalman_measurement_noise_y)),
                max_predict_missing_ms=max(1.0, float(kalman_max_predict_missing_ms)),
                max_predict_steps=max(1, int(kalman_max_predict_steps)),
                max_predict_dt_ms=max(1.0, float(kalman_max_predict_dt_ms)),
                max_position_sigma_px=max(1.0, float(kalman_max_position_sigma_px)),
                max_covariance_trace=max(1.0, float(kalman_max_covariance_trace)),
                nis_threshold=max(1e-6, float(kalman_nis_threshold)),
                nis_hard_reject=max(1e-6, float(kalman_nis_hard_reject)),
                min_identity_confidence=max(0.0, min(1.0, float(kalman_min_identity_confidence))),
                min_prediction_confidence=max(0.0, min(1.0, float(kalman_min_prediction_confidence))),
                prediction_decay_tau_ms=max(1.0, float(kalman_prediction_decay_tau_ms)),
            ),
        )
        filter_result = BasicCandidateFilter().apply(
            context,
            allowed_class_ids=parse_allowed_class_ids(class_filter),
            min_confidence=max(0.0, min(1.0, float(min_confidence))),
            aim_y_ratio=max(0.0, min(1.0, float(aim_ratio))),
            class_aim_y_ratios=class_aim_y_ratios,
        )
        basic_filter_debug = filter_result.debug_payload()
        center_x = (
            float(control_center_x_px)
            if control_center_x_px is not None
            else float(context.width) * 0.5
        )
        center_y = (
            float(control_center_y_px)
            if control_center_y_px is not None
            else float(context.height) * 0.5
        )
        fov_radius_px = min(float(context.width), float(context.height)) * max(
            0.0,
            min(1.0, float(fov_ratio)),
        )
        association_filter_result = AssociationCandidateFilter().apply(
            context,
            filter_result.observations,
            center_x_px=center_x,
            center_y_px=center_y,
            radius_px=fov_radius_px,
            max_aspect_ratio=ratio_max_aspect,
        )
        association_filter_debug = association_filter_result.debug_payload()
        filter_debug = {
            **association_filter_debug,
            "basic_filter": basic_filter_debug,
            "association_filter": association_filter_debug,
        }
        tracker_update = self.tracker.update(
            association_filter_result.observations,
            int(context.capture_ts_ns or 0),
            frame_id=context.frame_id,
        )
        if not tracker_update.tracks:
            self._lost_count += 1 if self._locked is not None else 0
            unavailable_reason = (
                self._first_association_rejection_reason(association_filter_result.rejected)
                if not association_filter_result.observations and association_filter_result.rejected
                else
                self._first_basic_rejection_reason(filter_result.rejected)
                if filter_result.rejected
                else tracker_update.reason
            )
            self.last_debug = {
                **filter_debug,
                "min_confidence": float(min_confidence),
                "class_filter": str(class_filter),
                "fov_ratio": float(fov_ratio),
                "fov_radius_px": fov_radius_px,
                "control_center_roi_px": {"x": center_x, "y": center_y},
                "reason": unavailable_reason,
                "tracker": tracker_update.debug,
            }
            return self._lost_or_clear(
                lost_grace_frames,
                unavailable_reason,
                0,
                0,
                state="no_target",
            )

        return self._select_from_tracker_update(
            context,
            tracker_update,
            filter_debug,
            min_confidence=min_confidence,
            fov_ratio=fov_ratio,
            aim_ratio=aim_ratio,
            control_center_x_px=control_center_x_px,
            control_center_y_px=control_center_y_px,
            class_filter=class_filter,
            ratio_max_aspect=ratio_max_aspect,
            quality_confidence_weight=quality_confidence_weight,
            quality_area_weight=quality_area_weight,
            class_priority=class_priority,
            selection_class_weight=selection_class_weight,
            selection_quality_weight=selection_quality_weight,
            selection_distance_weight=selection_distance_weight,
            sticky_bias=sticky_bias,
            lock_enabled=lock_enabled,
            lost_grace_frames=lost_grace_frames,
            target_switch_min_preference_advantage=target_switch_min_preference_advantage,
            target_switch_min_continuity_score=target_switch_min_continuity_score,
            target_switch_delay_ms=target_switch_delay_ms,
        )

    def _select_from_tracker_update(
        self,
        context: FrameContext,
        tracker_update: TrackerUpdate,
        filter_debug: dict,
        *,
        min_confidence: float,
        fov_ratio: float,
        aim_ratio: float,
        control_center_x_px: float | None,
        control_center_y_px: float | None,
        class_filter: str,
        ratio_max_aspect: float,
        quality_confidence_weight: float,
        quality_area_weight: float,
        class_priority: Iterable[int],
        selection_class_weight: float,
        selection_quality_weight: float,
        selection_distance_weight: float,
        sticky_bias: float,
        lock_enabled: bool,
        lost_grace_frames: int,
        target_switch_min_preference_advantage: float,
        target_switch_min_continuity_score: float,
        target_switch_delay_ms: float,
    ) -> TargetSelection:
        center_x = (
            float(control_center_x_px)
            if control_center_x_px is not None
            else context.width / 2
        )
        center_y = (
            float(control_center_y_px)
            if control_center_y_px is not None
            else context.height / 2
        )
        fov_radius_px = min(float(context.width), float(context.height)) * max(
            0.0,
            min(1.0, float(fov_ratio)),
        )
        candidates = list(tracker_update.tracks)
        priority = {int(cls): rank for rank, cls in enumerate(class_priority)}
        locked = self._locked_match(candidates) if lock_enabled else None
        track_filter_result = self._score_confirmed_tracks(
            context,
            candidates,
            center_x=center_x,
            center_y=center_y,
            fov_radius_px=fov_radius_px,
            priority=priority,
            locked=locked,
            sticky_bias=sticky_bias,
            quality_confidence_weight=quality_confidence_weight,
            quality_area_weight=quality_area_weight,
            selection_class_weight=selection_class_weight,
            selection_quality_weight=selection_quality_weight,
            selection_distance_weight=selection_distance_weight,
        )
        scored_candidates = track_filter_result.tracks
        candidates = [item.track for item in scored_candidates]
        if not scored_candidates:
            self._lost_count += 1 if self._locked is not None else 0
            self.last_debug = {
                **filter_debug,
                "tracked_filter": track_filter_result.debug_payload(),
                "min_confidence": float(min_confidence),
                "class_filter": str(class_filter),
                "fov_ratio": float(fov_ratio),
                "fov_radius_px": fov_radius_px,
                "control_center_roi_px": {"x": center_x, "y": center_y},
                "reason": "no confirmed track after candidate filter",
                "tracker": tracker_update.debug,
            }
            return self._lost_or_clear(
                lost_grace_frames,
                "no confirmed track after candidate filter",
                0,
                0,
                state="no_target",
            )
        aim_ratio = max(0.0, min(1.0, float(aim_ratio)))
        scored_by_id = {id(item.track): item for item in scored_candidates}
        best_scored = max(
            scored_candidates,
            key=lambda item: (
                item.selection_score,
                item.class_score,
                item.distance_score,
                item.quality.quality_score,
                float(item.track.score),
                -int(item.track.track_id),
            ),
        )
        best = best_scored.track
        previous_key = self._locked.key if self._locked is not None else None
        best_key = self._target_key(best)
        selected_locked = locked is not None and self._target_key(best) == self._target_key(locked)
        best_continuity = self._track_continuity_score(tracker_update, best)
        switch_debug = self._switch_debug_payload()
        self.last_debug = {
            **filter_debug,
            "tracked_filter": track_filter_result.debug_payload(),
            "fov_ratio": float(fov_ratio),
            "fov_radius_px": fov_radius_px,
            "aim_ratio": float(aim_ratio),
            "control_center_roi_px": {"x": center_x, "y": center_y},
            "min_confidence": float(min_confidence),
            "class_filter": str(class_filter),
            "ratio_max_aspect": float(ratio_max_aspect),
            "quality_confidence_weight": float(quality_confidence_weight),
            "quality_area_weight": float(quality_area_weight),
            "selection_class_weight": float(selection_class_weight),
            "selection_quality_weight": float(selection_quality_weight),
            "selection_distance_weight": float(selection_distance_weight),
            "best_selection_score": float(best_scored.selection_score),
            "sticky_bias": float(sticky_bias),
            "lock_enabled": bool(lock_enabled),
            "tracker": tracker_update.debug,
            "selected": scored_track_debug(best_scored),
            "selected_key": best_key,
            "locked_key": previous_key,
            "selected_continuity_score": best_continuity,
            "switch": switch_debug,
        }
        if self._locked is None or not lock_enabled:
            self._pending_switch = None
            self._remember(best)
            self.last_debug["switch"] = {
                "state": "committed_initial" if previous_key is None else "lock_disabled",
                "committed": True,
                "from_key": previous_key,
                "to_key": best_key,
                "reason": "initial target committed" if previous_key is None else "target lock disabled",
            }
            return TargetSelection(
                target=best,
                state="fresh" if previous_key == best_key else "acquire",
                reason="initial target committed" if previous_key is None else "target lock disabled",
                candidates=len(candidates),
                inside_fov=track_filter_result.inside_fov_count,
                locked=selected_locked,
                priority_rank=self._priority_rank(best, priority),
                distance_px=self._aim_distance(best, center_x, center_y),
                quality_score=best_scored.quality.quality_score,
                class_score=best_scored.class_score,
                distance_score=best_scored.distance_score,
                selection_score=best_scored.selection_score,
            )

        if previous_key != best_key:
            switch = self._advance_switch(
                candidate=best,
                candidate_score=best_scored,
                locked_score=scored_by_id.get(id(locked)) if locked is not None else None,
                tracker_update=tracker_update,
                min_preference_advantage=target_switch_min_preference_advantage,
                min_continuity_score=target_switch_min_continuity_score,
                delay_ms=target_switch_delay_ms,
                now_ns=_context_time_ns(context),
            )
            self.last_debug["switch"] = switch
            if not switch["committed"]:
                if locked is not None:
                    locked_scored = scored_by_id.get(id(locked))
                    self._remember(locked)
                    if locked_scored is not None:
                        self.last_debug["selected"] = scored_track_debug(locked_scored)
                    self.last_debug["selected_key"] = self._target_key(locked)
                    return TargetSelection(
                        target=locked,
                        state="switch_hold",
                        reason=switch["reason"],
                        candidates=len(candidates),
                        inside_fov=track_filter_result.inside_fov_count,
                        locked=True,
                        lost_count=self._lost_count,
                        priority_rank=self._priority_rank(locked, priority),
                        distance_px=self._aim_distance(
                            locked,
                            center_x,
                            center_y,
                        ),
                        quality_score=(
                            locked_scored.quality.quality_score
                            if locked_scored is not None
                            else None
                        ),
                        class_score=locked_scored.class_score if locked_scored is not None else None,
                        distance_score=(
                            locked_scored.distance_score if locked_scored is not None else None
                        ),
                        selection_score=(
                            locked_scored.selection_score if locked_scored is not None else None
                        ),
                    )
                return TargetSelection(
                    target=None,
                    state="switch_pending",
                    reason=switch["reason"],
                    candidates=len(candidates),
                    inside_fov=track_filter_result.inside_fov_count,
                    locked=True,
                    lost_count=self._lost_count,
                    priority_rank=self._priority_rank(best, priority),
                    distance_px=self._aim_distance(best, center_x, center_y),
                    quality_score=best_scored.quality.quality_score,
                    class_score=best_scored.class_score,
                    distance_score=best_scored.distance_score,
                    selection_score=best_scored.selection_score,
                )
            self._remember(best)
            self.last_debug["selected"] = scored_track_debug(best_scored)
            return TargetSelection(
                target=best,
                state="switch_committed",
                reason=switch["reason"],
                candidates=len(candidates),
                inside_fov=track_filter_result.inside_fov_count,
                locked=False,
                priority_rank=self._priority_rank(best, priority),
                distance_px=self._aim_distance(best, center_x, center_y),
                quality_score=best_scored.quality.quality_score,
                class_score=best_scored.class_score,
                distance_score=best_scored.distance_score,
                selection_score=best_scored.selection_score,
            )

        self._pending_switch = None
        self._remember(best)
        self.last_debug["switch"] = {
            "state": "same_target",
            "committed": False,
            "from_key": previous_key,
            "to_key": best_key,
            "reason": "same locked target",
        }
        selection_state = "locked" if selected_locked else "fresh"
        selection_reason = "按类别、候选质量、归一化距离综合分和锁定切换门控选择目标"
        return TargetSelection(
            target=best,
            state=selection_state,
            reason=selection_reason,
            candidates=len(candidates),
            inside_fov=track_filter_result.inside_fov_count,
            locked=selected_locked,
            priority_rank=self._priority_rank(best, priority),
            distance_px=self._aim_distance(best, center_x, center_y),
            quality_score=best_scored.quality.quality_score,
            class_score=best_scored.class_score,
            distance_score=best_scored.distance_score,
            selection_score=best_scored.selection_score,
        )

    def _score_confirmed_tracks(
        self,
        context: FrameContext,
        candidates: list[Track],
        *,
        center_x: float,
        center_y: float,
        fov_radius_px: float,
        priority: dict[int, int],
        locked: Track | None,
        sticky_bias: float,
        quality_confidence_weight: float,
        quality_area_weight: float,
        selection_class_weight: float,
        selection_quality_weight: float,
        selection_distance_weight: float,
    ) -> TrackScoreResult:
        quality = QualityScorer(
            QualityScoreConfig(
                confidence_weight=max(0.0, float(quality_confidence_weight)),
                area_weight=max(0.0, float(quality_area_weight)),
            )
        )
        class_weight = max(0.0, float(selection_class_weight))
        quality_weight = max(0.0, float(selection_quality_weight))
        distance_weight = max(0.0, float(selection_distance_weight))
        total_weight = class_weight + quality_weight + distance_weight
        if total_weight <= 0.0:
            quality_weight = 1.0
            total_weight = 1.0
        radius = max(1e-6, float(fov_radius_px))
        class_areas: dict[int, list[float]] = {}
        for track in candidates:
            class_areas.setdefault(int(track.cls), []).append(max(1.0, float(track.area)))
        class_reference_areas = {
            class_id: float(median(areas))
            for class_id, areas in class_areas.items()
        }
        scored: list[ScoredTrack] = []
        for track in candidates:
            aim_x, aim_y = track.filtered_aim_px
            base_quality = quality.score(
                track,
                context=context,
                class_reference_area=class_reference_areas.get(int(track.cls)),
            )
            candidate_quality = CandidateQuality(
                conf_score=base_quality.conf_score,
                area_score=base_quality.area_score,
                quality_score=min(
                    base_quality.quality_score,
                    max(0.0, min(1.0, float(track.quality_score))),
                ),
            )
            distance_px = self._selection_distance(
                track,
                locked,
                sticky_bias,
                center_x,
                center_y,
            )
            distance_score = 1.0 - max(0.0, min(1.0, distance_px / radius))
            class_score = self._class_preference_score(track, priority)
            selection_score = (
                class_weight * class_score
                + quality_weight * candidate_quality.quality_score
                + distance_weight * distance_score
            ) / total_weight
            scored.append(
                ScoredTrack(
                    frame_id=int(context.frame_id),
                    capture_ts_ns=context.capture_ts_ns,
                    track=track,
                    quality=candidate_quality,
                    distance_px=self._aim_distance(track, center_x, center_y),
                    aim_x=float(aim_x),
                    aim_y=float(aim_y),
                    class_score=class_score,
                    distance_score=distance_score,
                    selection_score=max(0.0, min(1.0, selection_score)),
                )
            )
        return TrackScoreResult(
            tracks=scored,
            raw_count=len(candidates),
        )

    @staticmethod
    def _first_basic_rejection_reason(rejected: list[dict]) -> str:
        reasons = [str(item.get("reason") or "") for item in rejected]
        if reasons and all(reason == "confidence_filter" for reason in reasons):
            return "no candidate after confidence filter"
        if reasons and all(reason == "class_filter" for reason in reasons):
            return "no candidate after class filter"
        if reasons and all(reason == "invalid_bbox" for reason in reasons):
            return "no candidate after bbox validation"
        return "no candidate after basic candidate filter"

    @staticmethod
    def _first_association_rejection_reason(rejected: list[dict]) -> str:
        reasons = [str(item.get("reason") or "") for item in rejected]
        if reasons and all(reason == "selection_fov" for reason in reasons):
            return "no candidate inside selection fov"
        if reasons and all(reason == "ratio_check" for reason in reasons):
            return "no candidate after ratio check"
        return "no candidate after association candidate filter"

    def _lost_or_clear(
        self,
        grace: int,
        reason: str,
        candidates: int,
        inside_fov: int,
        *,
        state: str | None = None,
    ) -> TargetSelection:
        self._pending_switch = None
        if self._locked is None:
            return TargetSelection(
                None,
                state or "no_target",
                reason,
                candidates=candidates,
                inside_fov=inside_fov,
                lost_count=0,
            )
        if self._lost_count <= max(0, grace):
            return TargetSelection(
                None,
                state or "lost",
                reason,
                candidates=candidates,
                inside_fov=inside_fov,
                locked=True,
                lost_count=self._lost_count,
            )
        lost_count = self._lost_count
        self._locked = None
        self._pending_switch = None
        self._lost_count = 0
        return TargetSelection(
            None,
            state or "reacquire",
            reason,
            candidates=candidates,
            inside_fov=inside_fov,
            lost_count=lost_count,
        )

    def _remember(self, target: Track) -> None:
        key = self._target_key(target)
        self._locked = _LockedTarget(key=key)
        self._lost_count = 0

    def _advance_switch(
        self,
        *,
        candidate: Track,
        candidate_score: ScoredTrack,
        locked_score: ScoredTrack | None,
        tracker_update: TrackerUpdate,
        min_preference_advantage: float,
        min_continuity_score: float,
        delay_ms: float,
        now_ns: int,
    ) -> dict:
        candidate_key = self._target_key(candidate)
        from_key = self._locked.key if self._locked is not None else None
        advantage = self._preference_advantage(candidate_score, locked_score)
        continuity = self._track_continuity_score(tracker_update, candidate)
        required_advantage = max(0.0, float(min_preference_advantage))
        required_continuity = max(0.0, min(1.0, float(min_continuity_score)))
        required_delay_ms = max(0.0, float(delay_ms))
        eligible = advantage >= required_advantage and continuity >= required_continuity
        if not eligible:
            self._pending_switch = None
            reason = (
                "switch pending rejected: "
                f"advantage={advantage:.3f}/{required_advantage:.3f}, "
                f"continuity={continuity:.3f}/{required_continuity:.3f}"
            )
            return {
                "state": "rejected",
                "committed": False,
                "from_key": from_key,
                "to_key": candidate_key,
                "elapsed_ms": 0.0,
                "required_delay_ms": required_delay_ms,
                "advantage": advantage,
                "required_advantage": required_advantage,
                "continuity_score": continuity,
                "required_continuity_score": required_continuity,
                "reason": reason,
            }
        previous = self._pending_switch
        started_ts_ns = (
            previous.started_ts_ns
            if previous is not None and previous.key == candidate_key
            else int(now_ns)
        )
        elapsed_ms = max(0.0, (int(now_ns) - started_ts_ns) / 1e6)
        reason = (
            "SWITCH_PENDING: "
            f"{elapsed_ms:.1f}/{required_delay_ms:.1f}ms, "
            f"advantage={advantage:.3f}, continuity={continuity:.3f}"
        )
        self._pending_switch = _PendingSwitch(
            key=candidate_key,
            started_ts_ns=started_ts_ns,
            elapsed_ms=elapsed_ms,
            advantage=advantage,
            continuity_score=continuity,
            reason=reason,
        )
        committed = elapsed_ms >= required_delay_ms
        if committed:
            self._pending_switch = None
            reason = (
                "SWITCH_COMMITTED: "
                f"{from_key} -> {candidate_key}; "
                f"elapsed_ms={elapsed_ms:.1f}, advantage={advantage:.3f}, continuity={continuity:.3f}"
            )
        return {
            "state": "committed" if committed else "pending",
            "committed": committed,
            "from_key": from_key,
            "to_key": candidate_key,
            "elapsed_ms": elapsed_ms,
            "required_delay_ms": required_delay_ms,
            "advantage": advantage,
            "required_advantage": required_advantage,
            "continuity_score": continuity,
            "required_continuity_score": required_continuity,
            "reason": reason,
        }

    def _switch_debug_payload(self) -> dict:
        if self._pending_switch is None:
            return {"state": "idle", "committed": False}
        return {
            "state": "pending",
            "committed": False,
            "to_key": self._pending_switch.key,
            "elapsed_ms": self._pending_switch.elapsed_ms,
            "advantage": self._pending_switch.advantage,
            "continuity_score": self._pending_switch.continuity_score,
            "reason": self._pending_switch.reason,
        }

    @staticmethod
    def _preference_advantage(
        candidate_score: ScoredTrack,
        locked_score: ScoredTrack | None,
    ) -> float:
        locked_value = float(locked_score.selection_score) if locked_score is not None else 0.0
        return float(candidate_score.selection_score) - locked_value

    @staticmethod
    def _track_continuity_score(tracker_update: TrackerUpdate, target: Track) -> float:
        track_id = target.track_id
        for item in tracker_update.debug.get("tracks", []):
            try:
                if int(item.get("track_id")) == int(track_id):
                    return max(0.0, min(1.0, float(item.get("identity_confidence", 0.0))))
            except (TypeError, ValueError):
                continue
        return 0.0

    def _locked_match(self, candidates: list[Track]) -> Track | None:
        if self._locked is None:
            return None
        for item in candidates:
            if self._target_key(item) == self._locked.key:
                return item
        return None

    def _selection_distance(
        self,
        target: Track,
        locked: Track | None,
        sticky_bias: float,
        center_x: float,
        center_y: float,
    ) -> float:
        distance = self._aim_distance(target, center_x, center_y)
        if locked is None or sticky_bias <= 0:
            return distance
        if self._target_key(target) != self._target_key(locked):
            return distance
        return distance * max(0.0, 1.0 - min(0.9, sticky_bias))

    @staticmethod
    def _target_key(target: Track) -> str:
        return f"track:{int(target.track_id)}"

    @staticmethod
    def _priority_rank(target: Track, priority: dict[int, int]) -> int:
        return priority.get(int(target.cls), len(priority) + int(target.cls))

    @staticmethod
    def _class_preference_score(target: Track, priority: dict[int, int]) -> float:
        rank = priority.get(int(target.cls))
        if rank == 0:
            return 1.0
        if rank == 1:
            return 0.5
        return 0.0

    @staticmethod
    def _aim_distance(target: Track, center_x: float, center_y: float) -> float:
        aim_x, aim_y = target.filtered_aim_px
        return ((float(aim_x) - center_x) ** 2 + (float(aim_y) - center_y) ** 2) ** 0.5
