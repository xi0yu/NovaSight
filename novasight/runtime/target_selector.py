from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from novasight.contracts import Detection, FrameContext, Track
from novasight.runtime.candidates import (
    CandidateFilter,
    CandidateFilterConfig,
    CandidateFilterResult,
    QualityScoreConfig,
    QualityScorer,
    RatioCheckConfig,
    ScoredCandidate,
    SelectionFovConfig,
    candidate_debug,
    parse_allowed_class_ids,
)
from novasight.runtime.kalman import KalmanConfig
from novasight.runtime.tracker import RuntimeTracker, TrackerConfig, TrackerUpdate


Target = Detection | Track


@dataclass(frozen=True)
class TargetSelection:
    target: Target | None
    state: str
    reason: str
    candidates: int = 0
    inside_fov: int = 0
    locked: bool = False
    lost_count: int = 0
    priority_rank: int | None = None
    distance_px: float | None = None
    quality_score: float | None = None


@dataclass
class _LockedTarget:
    key: str
    cx: float
    cy: float
    cls: int


@dataclass
class _PendingSwitch:
    key: str
    frames: int
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
        self.tracker.reset()

    def select(
        self,
        context: FrameContext,
        *,
        min_confidence: float,
        fov_ratio: float,
        aim_ratio: float = 50.0,
        class_filter: str = "all",
        class_priority: Iterable[int] = (),
        sticky_bias: float = 0.25,
        lock_enabled: bool = True,
        lost_grace_frames: int = 5,
        ratio_max_aspect: float = 6.0,
        quality_confidence_weight: float = 0.7,
        quality_area_weight: float = 0.3,
        class_priority_quality_margin: float = 0.08,
        tracker_confirm_frames: int = 2,
        tracker_matching_distance_px: float = 140.0,
        tracker_ambiguity_margin: float = 0.08,
        tracker_missing_timeout_ms: float = 120.0,
        tracker_delete_timeout_ms: float = 250.0,
        tracker_match_threshold: float = 0.65,
        tracker_mahalanobis_gate: float = 9.21,
        kalman_enabled: bool = True,
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
        target_switch_confirm_frames: int = 3,
    ) -> TargetSelection:
        raw_candidates: list[Target] = list(context.tracks) or list(context.detections)
        if context.width <= 0 or context.height <= 0:
            self._lost_count += 1 if self._locked is not None else 0
            tracker_update = self.tracker.update(context, [])
            self.last_debug = {
                "raw_candidates": len(raw_candidates),
                "filtered_candidates": 0,
                "inside_fov": 0,
                "reason": "invalid frame geometry",
                "tracker": tracker_update.debug,
            }
            return self._lost_or_clear(lost_grace_frames, "invalid frame geometry", 0, 0)

        self.tracker.config = TrackerConfig(
            confirm_frames=max(1, int(tracker_confirm_frames)),
            matching_distance_px=max(1.0, float(tracker_matching_distance_px)),
            ambiguity_margin=max(0.0, float(tracker_ambiguity_margin)),
            missing_timeout_ms=max(1.0, float(tracker_missing_timeout_ms)),
            delete_timeout_ms=max(1.0, float(tracker_delete_timeout_ms)),
            match_threshold=max(0.0, min(1.0, float(tracker_match_threshold))),
            mahalanobis_gate=max(1e-6, float(tracker_mahalanobis_gate)),
            kalman=KalmanConfig(
                enabled=bool(kalman_enabled),
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
        filter_result = self._filter_candidates(
            context,
            raw_candidates,
            min_confidence=min_confidence,
            fov_ratio=fov_ratio,
            aim_ratio=aim_ratio,
            class_filter=class_filter,
            ratio_max_aspect=ratio_max_aspect,
            quality_confidence_weight=quality_confidence_weight,
            quality_area_weight=quality_area_weight,
        )
        scored_candidates = filter_result.candidates
        filter_debug = filter_result.debug_payload()
        if not scored_candidates:
            tracker_update = self._update_tracker_without_candidates(context, filter_result)
            if tracker_update.tracks:
                return self._select_from_tracker_update(
                    context,
                    tracker_update,
                    filter_debug,
                    min_confidence=min_confidence,
                    fov_ratio=fov_ratio,
                    aim_ratio=aim_ratio,
                    class_filter=class_filter,
                    ratio_max_aspect=ratio_max_aspect,
                    quality_confidence_weight=quality_confidence_weight,
                    quality_area_weight=quality_area_weight,
                    class_priority=class_priority,
                    sticky_bias=sticky_bias,
                    lock_enabled=lock_enabled,
                    lost_grace_frames=lost_grace_frames,
                    class_priority_quality_margin=class_priority_quality_margin,
                    target_switch_min_preference_advantage=target_switch_min_preference_advantage,
                    target_switch_min_continuity_score=target_switch_min_continuity_score,
                    target_switch_confirm_frames=target_switch_confirm_frames,
                )
            self._lost_count += 1 if self._locked is not None else 0
            self.last_debug = {
                **filter_debug,
                "min_confidence": float(min_confidence),
                "class_filter": str(class_filter),
                "reason": self._first_rejection_reason(filter_result),
                "tracker": tracker_update.debug,
            }
            return self._lost_or_clear(
                lost_grace_frames,
                self._first_rejection_reason(filter_result),
                0,
                0,
                state=tracker_update.state.lower(),
            )

        tracker_update = self.tracker.update(context, scored_candidates)
        if not tracker_update.tracks:
            self._lost_count += 1 if self._locked is not None else 0
            self.last_debug = {
                **filter_debug,
                "min_confidence": float(min_confidence),
                "class_filter": str(class_filter),
                "reason": tracker_update.reason,
                "tracker": tracker_update.debug,
            }
            return self._lost_or_clear(
                lost_grace_frames,
                tracker_update.reason,
                0,
                0,
                state=tracker_update.state.lower(),
            )

        return self._select_from_tracker_update(
            context,
            tracker_update,
            filter_debug,
            min_confidence=min_confidence,
            fov_ratio=fov_ratio,
            aim_ratio=aim_ratio,
            class_filter=class_filter,
            ratio_max_aspect=ratio_max_aspect,
            quality_confidence_weight=quality_confidence_weight,
            quality_area_weight=quality_area_weight,
            class_priority=class_priority,
            sticky_bias=sticky_bias,
            lock_enabled=lock_enabled,
            lost_grace_frames=lost_grace_frames,
            class_priority_quality_margin=class_priority_quality_margin,
            target_switch_min_preference_advantage=target_switch_min_preference_advantage,
            target_switch_min_continuity_score=target_switch_min_continuity_score,
            target_switch_confirm_frames=target_switch_confirm_frames,
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
        class_filter: str,
        ratio_max_aspect: float,
        quality_confidence_weight: float,
        quality_area_weight: float,
        class_priority: Iterable[int],
        sticky_bias: float,
        lock_enabled: bool,
        lost_grace_frames: int,
        class_priority_quality_margin: float,
        target_switch_min_preference_advantage: float,
        target_switch_min_continuity_score: float,
        target_switch_confirm_frames: int,
    ) -> TargetSelection:
        track_filter_result = self._filter_candidates(
            context,
            list(tracker_update.tracks),
            min_confidence=min_confidence,
            fov_ratio=fov_ratio,
            aim_ratio=aim_ratio,
            class_filter=class_filter,
            ratio_max_aspect=ratio_max_aspect,
            quality_confidence_weight=quality_confidence_weight,
            quality_area_weight=quality_area_weight,
        )
        scored_candidates = track_filter_result.candidates
        candidates = [item.detection for item in scored_candidates]
        if not scored_candidates:
            self._lost_count += 1 if self._locked is not None else 0
            self.last_debug = {
                **filter_debug,
                "tracked_filter": track_filter_result.debug_payload(),
                "min_confidence": float(min_confidence),
                "class_filter": str(class_filter),
                "reason": "no confirmed track after candidate filter",
                "tracker": tracker_update.debug,
            }
            return self._lost_or_clear(
                lost_grace_frames,
                "no confirmed track after candidate filter",
                0,
                0,
                state=tracker_update.state.lower(),
            )

        center_x = context.width / 2
        center_y = context.height / 2
        aim_ratio = max(0.0, min(100.0, float(aim_ratio)))
        priority = {int(cls): rank for rank, cls in enumerate(class_priority)}
        locked = self._locked_match(candidates) if lock_enabled else None
        scored_by_id = {id(item.detection): item for item in scored_candidates}
        quality_margin = max(0.0, float(class_priority_quality_margin))
        best_quality = max(item.quality.quality_score for item in scored_candidates)
        viable_scored = [
            item for item in scored_candidates
            if item.quality.quality_score >= best_quality - quality_margin
        ]
        viable = [item.detection for item in viable_scored]

        best = min(
            viable,
            key=lambda item: (
                self._priority_rank(item, priority),
                self._selection_distance(item, locked, sticky_bias, center_x, center_y, aim_ratio),
                -scored_by_id[id(item)].quality.quality_score,
                -float(item.score),
            ),
        )
        best_scored = scored_by_id[id(best)]
        previous_key = self._locked.key if self._locked is not None else None
        best_key = self._target_key(best)
        selected_locked = locked is not None and self._target_key(best) == self._target_key(locked)
        best_continuity = self._track_continuity_score(tracker_update, best)
        switch_debug = self._switch_debug_payload()
        self.last_debug = {
            **filter_debug,
            "tracked_filter": track_filter_result.debug_payload(),
            "fov_ratio": float(fov_ratio),
            "aim_ratio": float(aim_ratio),
            "min_confidence": float(min_confidence),
            "class_filter": str(class_filter),
            "ratio_max_aspect": float(ratio_max_aspect),
            "quality_confidence_weight": float(quality_confidence_weight),
            "quality_area_weight": float(quality_area_weight),
            "class_priority_quality_margin": float(class_priority_quality_margin),
            "best_quality": float(best_quality),
            "sticky_bias": float(sticky_bias),
            "lock_enabled": bool(lock_enabled),
            "tracker": tracker_update.debug,
            "selected": candidate_debug(best_scored),
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
                distance_px=self._aim_distance(best, center_x, center_y, aim_ratio),
                quality_score=best_scored.quality.quality_score,
            )

        if previous_key != best_key:
            switch = self._advance_switch(
                candidate=best,
                candidate_score=best_scored,
                locked=locked,
                locked_score=scored_by_id.get(id(locked)) if locked is not None else None,
                priority=priority,
                quality_margin=quality_margin,
                tracker_update=tracker_update,
                min_preference_advantage=target_switch_min_preference_advantage,
                min_continuity_score=target_switch_min_continuity_score,
                confirm_frames=target_switch_confirm_frames,
            )
            self.last_debug["switch"] = switch
            if not switch["committed"]:
                return TargetSelection(
                    target=None,
                    state="switch_pending",
                    reason=switch["reason"],
                    candidates=len(candidates),
                    inside_fov=track_filter_result.inside_fov_count,
                    locked=True,
                    lost_count=self._lost_count,
                    priority_rank=self._priority_rank(best, priority),
                    distance_px=self._aim_distance(best, center_x, center_y, aim_ratio),
                    quality_score=best_scored.quality.quality_score,
                )
            self._remember(best)
            self.last_debug["selected"] = candidate_debug(best_scored)
            return TargetSelection(
                target=best,
                state="switch_committed",
                reason=switch["reason"],
                candidates=len(candidates),
                inside_fov=track_filter_result.inside_fov_count,
                locked=False,
                priority_rank=self._priority_rank(best, priority),
                distance_px=self._aim_distance(best, center_x, center_y, aim_ratio),
                quality_score=best_scored.quality.quality_score,
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
        selection_state = (
            "predicting"
            if tracker_update.state == "PREDICTING"
            else "locked" if selected_locked else "fresh"
        )
        selection_reason = (
            "valid Kalman prediction within limited prediction window"
            if tracker_update.state == "PREDICTING"
            else "按质量候选、类别优先级、距离和锁定偏好选择目标"
        )
        return TargetSelection(
            target=best,
            state=selection_state,
            reason=selection_reason,
            candidates=len(candidates),
            inside_fov=track_filter_result.inside_fov_count,
            locked=selected_locked,
            priority_rank=self._priority_rank(best, priority),
            distance_px=self._aim_distance(best, center_x, center_y, aim_ratio),
            quality_score=best_scored.quality.quality_score,
        )

    def _update_tracker_without_candidates(
        self,
        context: FrameContext,
        filter_result: CandidateFilterResult,
    ) -> TrackerUpdate:
        if any(item.reason == "selection_fov" for item in filter_result.rejected):
            return self.tracker.mark_unavailable(
                "OUT_OF_ROI",
                context=context,
                reason="candidate outside selection fov",
            )
        return self.tracker.update(context, [])

    def _filter_candidates(
        self,
        context: FrameContext,
        candidates: list[Target],
        *,
        min_confidence: float,
        fov_ratio: float,
        aim_ratio: float,
        class_filter: str,
        ratio_max_aspect: float,
        quality_confidence_weight: float,
        quality_area_weight: float,
    ) -> CandidateFilterResult:
        radius_percent = max(0.0, min(100.0, float(fov_ratio) * 100.0))
        config = CandidateFilterConfig(
            allowed_class_ids=parse_allowed_class_ids(class_filter),
            min_confidence=max(0.0, min(1.0, float(min_confidence))),
            selection_fov=SelectionFovConfig(
                enabled=True,
                shape="circle",
                radius_x_percent=radius_percent,
                radius_y_percent=radius_percent,
            ),
            ratio_check=RatioCheckConfig(max_aspect_ratio=max(1.0, float(ratio_max_aspect))),
        )
        quality = QualityScorer(
            QualityScoreConfig(
                confidence_weight=max(0.0, float(quality_confidence_weight)),
                area_weight=max(0.0, float(quality_area_weight)),
            )
        )
        return CandidateFilter(config, quality_scorer=quality).apply(
            context,
            candidates,
            aim_ratio=aim_ratio,
        )

    @staticmethod
    def _first_rejection_reason(result: CandidateFilterResult) -> str:
        if not result.rejected:
            return "no candidate after candidate filter"
        reasons = [item.reason for item in result.rejected]
        if all(reason == "selection_fov" for reason in reasons):
            return "no candidate inside selection fov"
        if all(reason == "confidence_filter" for reason in reasons):
            return "no candidate after confidence filter"
        if all(reason == "class_filter" for reason in reasons):
            return "no candidate after class filter"
        if all(reason == "ratio_check" for reason in reasons):
            return "no candidate after ratio check"
        if all(reason == "invalid_bbox" for reason in reasons):
            return "no candidate after bbox validation"
        return "no candidate after candidate filter"

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
        self.reset()
        return TargetSelection(
            None,
            state or "reacquire",
            reason,
            candidates=candidates,
            inside_fov=inside_fov,
            lost_count=lost_count,
        )

    def _remember(self, target: Target) -> None:
        self._locked = _LockedTarget(
            key=self._target_key(target),
            cx=float(target.cx),
            cy=float(target.cy),
            cls=int(target.cls),
        )
        self._lost_count = 0

    def _advance_switch(
        self,
        *,
        candidate: Target,
        candidate_score: ScoredCandidate,
        locked: Target | None,
        locked_score: ScoredCandidate | None,
        priority: dict[int, int],
        quality_margin: float,
        tracker_update: TrackerUpdate,
        min_preference_advantage: float,
        min_continuity_score: float,
        confirm_frames: int,
    ) -> dict:
        candidate_key = self._target_key(candidate)
        from_key = self._locked.key if self._locked is not None else None
        advantage = self._preference_advantage(
            candidate,
            candidate_score,
            locked,
            locked_score,
            priority=priority,
            quality_margin=quality_margin,
        )
        continuity = self._track_continuity_score(tracker_update, candidate)
        required_advantage = max(0.0, float(min_preference_advantage))
        required_continuity = max(0.0, min(1.0, float(min_continuity_score)))
        required_frames = max(1, int(confirm_frames))
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
                "frames": 0,
                "required_frames": required_frames,
                "advantage": advantage,
                "required_advantage": required_advantage,
                "continuity_score": continuity,
                "required_continuity_score": required_continuity,
                "reason": reason,
            }
        previous = self._pending_switch
        frames = previous.frames + 1 if previous is not None and previous.key == candidate_key else 1
        reason = (
            "SWITCH_PENDING: "
            f"{frames}/{required_frames} frames, "
            f"advantage={advantage:.3f}, continuity={continuity:.3f}"
        )
        self._pending_switch = _PendingSwitch(
            key=candidate_key,
            frames=frames,
            advantage=advantage,
            continuity_score=continuity,
            reason=reason,
        )
        committed = frames >= required_frames
        if committed:
            self._pending_switch = None
            reason = (
                "SWITCH_COMMITTED: "
                f"{from_key} -> {candidate_key}; "
                f"frames={frames}, advantage={advantage:.3f}, continuity={continuity:.3f}"
            )
        return {
            "state": "committed" if committed else "pending",
            "committed": committed,
            "from_key": from_key,
            "to_key": candidate_key,
            "frames": frames,
            "required_frames": required_frames,
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
            "frames": self._pending_switch.frames,
            "advantage": self._pending_switch.advantage,
            "continuity_score": self._pending_switch.continuity_score,
            "reason": self._pending_switch.reason,
        }

    def _preference_advantage(
        self,
        candidate: Target,
        candidate_score: ScoredCandidate,
        locked: Target | None,
        locked_score: ScoredCandidate | None,
        *,
        priority: dict[int, int],
        quality_margin: float,
    ) -> float:
        candidate_quality = float(candidate_score.quality.quality_score)
        locked_quality = float(locked_score.quality.quality_score) if locked_score is not None else 0.0
        advantage = candidate_quality - locked_quality
        if locked is None:
            return advantage
        candidate_rank = self._priority_rank(candidate, priority)
        locked_rank = self._priority_rank(locked, priority)
        if candidate_rank < locked_rank:
            advantage += max(0.0, float(quality_margin))
        elif candidate_rank > locked_rank:
            advantage -= max(0.0, float(quality_margin))
        return advantage

    @staticmethod
    def _track_continuity_score(tracker_update: TrackerUpdate, target: Target) -> float:
        track_id = getattr(target, "track_id", None)
        if track_id is None:
            return 1.0
        for item in tracker_update.debug.get("tracks", []):
            try:
                if int(item.get("track_id")) == int(track_id):
                    return max(0.0, min(1.0, float(item.get("identity_confidence", 0.0))))
            except (TypeError, ValueError):
                continue
        return 0.0

    def _locked_match(self, candidates: list[Target]) -> Target | None:
        if self._locked is None:
            return None
        for item in candidates:
            if self._target_key(item) == self._locked.key:
                return item
        nearest_same_class = [
            item for item in candidates
            if getattr(item, "track_id", None) is None and int(item.cls) == self._locked.cls
        ]
        if not nearest_same_class:
            return None
        nearest = min(
            nearest_same_class,
            key=lambda item: ((float(item.cx) - self._locked.cx) ** 2 + (float(item.cy) - self._locked.cy) ** 2) ** 0.5,
        )
        distance = ((float(nearest.cx) - self._locked.cx) ** 2 + (float(nearest.cy) - self._locked.cy) ** 2) ** 0.5
        return nearest if distance <= 96.0 else None

    def _selection_distance(
        self,
        target: Target,
        locked: Target | None,
        sticky_bias: float,
        center_x: float,
        center_y: float,
        aim_ratio: float,
    ) -> float:
        distance = self._aim_distance(target, center_x, center_y, aim_ratio)
        if locked is None or sticky_bias <= 0:
            return distance
        if self._target_key(target) != self._target_key(locked):
            return distance
        return distance * max(0.0, 1.0 - min(0.9, sticky_bias))

    @staticmethod
    def _target_key(target: Target) -> str:
        track_id = getattr(target, "track_id", None)
        if track_id is not None:
            return f"track:{int(track_id)}"
        return f"class:{int(target.cls)}:{round(float(target.cx), 1)}:{round(float(target.cy), 1)}"

    @staticmethod
    def _class_allowed(target: Target, class_filter: str) -> bool:
        if class_filter == "all":
            return True
        try:
            return int(target.cls) == int(class_filter)
        except ValueError:
            return True

    @staticmethod
    def _priority_rank(target: Target, priority: dict[int, int]) -> int:
        return priority.get(int(target.cls), len(priority) + int(target.cls))

    @staticmethod
    def _distance(target: Target, center_x: float, center_y: float) -> float:
        return ((float(target.cx) - center_x) ** 2 + (float(target.cy) - center_y) ** 2) ** 0.5

    @staticmethod
    def _aim_y(target: Target, aim_ratio: float) -> float:
        point_y = getattr(target, "point_y", None)
        ratio = max(0.0, min(100.0, aim_ratio)) / 100.0
        return float(point_y(ratio)) if callable(point_y) else float(target.y) + float(target.h) * ratio

    def _aim_distance(self, target: Target, center_x: float, center_y: float, aim_ratio: float) -> float:
        return ((float(target.cx) - center_x) ** 2 + (self._aim_y(target, aim_ratio) - center_y) ** 2) ** 0.5

    def _candidate_debug(self, candidates: list[Target], context: FrameContext, *, aim_ratio: float) -> list[dict]:
        center_x = context.width / 2
        center_y = context.height / 2
        return [
            {
                "cls": int(getattr(item, "cls", -1)),
                "score": float(getattr(item, "score", 0.0)),
                "cx": float(getattr(item, "cx", 0.0)),
                "cy": float(getattr(item, "cy", 0.0)),
                "distance_px": self._distance(item, center_x, center_y),
                "aim_x": float(getattr(item, "cx", 0.0)),
                "aim_y": self._aim_y(item, aim_ratio),
                "aim_distance_px": self._aim_distance(item, center_x, center_y, aim_ratio),
            }
            for item in candidates[:12]
        ]
