from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from novasight.contracts import Detection, FrameContext, Track


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


@dataclass
class _LockedTarget:
    key: str
    cx: float
    cy: float
    cls: int


class RuntimeTargetSelector:
    def __init__(self) -> None:
        self._locked: _LockedTarget | None = None
        self._lost_count = 0
        self.last_debug: dict = {}

    def reset(self) -> None:
        self._locked = None
        self._lost_count = 0

    def select(
        self,
        context: FrameContext,
        *,
        min_confidence: float,
        fov_ratio: float,
        class_filter: str = "all",
        class_priority: Iterable[int] = (),
        sticky_bias: float = 0.25,
        lock_enabled: bool = True,
        lost_grace_frames: int = 5,
    ) -> TargetSelection:
        candidates: list[Target] = list(context.tracks) or list(context.detections)
        raw_candidates = list(candidates)
        if not candidates or context.width <= 0 or context.height <= 0:
            self._lost_count += 1 if self._locked is not None else 0
            self.last_debug = {
                "raw_candidates": len(raw_candidates),
                "filtered_candidates": 0,
                "inside_fov": 0,
                "reason": "no target candidates",
            }
            return self._lost_or_clear(lost_grace_frames, "no target candidates", 0, 0)

        candidates = [
            item for item in candidates
            if float(item.score) >= min_confidence and self._class_allowed(item, class_filter)
        ]
        if not candidates:
            self._lost_count += 1 if self._locked is not None else 0
            self.last_debug = {
                "raw_candidates": len(raw_candidates),
                "filtered_candidates": 0,
                "inside_fov": 0,
                "min_confidence": float(min_confidence),
                "class_filter": str(class_filter),
                "reason": "no candidate after confidence/class filter",
                "raw": self._candidate_debug(raw_candidates, context),
            }
            return self._lost_or_clear(
                lost_grace_frames,
                "no candidate after confidence/class filter",
                0,
                0,
            )

        center_x = context.width / 2
        center_y = context.height / 2
        radius = min(context.width, context.height) * max(0.0, min(1.0, fov_ratio))
        inside_fov = [
            item
            for item in candidates
            if self._distance(item, center_x, center_y) <= radius
        ]
        if not inside_fov:
            self._lost_count += 1 if self._locked is not None else 0
            self.last_debug = {
                "raw_candidates": len(raw_candidates),
                "filtered_candidates": len(candidates),
                "inside_fov": 0,
                "fov_radius": float(radius),
                "fov_ratio": float(fov_ratio),
                "min_confidence": float(min_confidence),
                "class_filter": str(class_filter),
                "reason": "no candidate inside fov",
                "candidates": self._candidate_debug(candidates, context),
            }
            return self._lost_or_clear(
                lost_grace_frames,
                "no candidate inside fov",
                len(candidates),
                0,
            )

        priority = {int(cls): rank for rank, cls in enumerate(class_priority)}
        locked = self._locked_match(inside_fov) if lock_enabled else None

        best = min(
            inside_fov,
            key=lambda item: (
                self._selection_distance(item, locked, sticky_bias, center_x, center_y),
                self._priority_rank(item, priority),
                -float(item.score),
            ),
        )
        previous_key = self._locked.key if self._locked is not None else None
        self._remember(best)
        selected_locked = locked is not None and self._target_key(best) == self._target_key(locked)
        self.last_debug = {
            "raw_candidates": len(raw_candidates),
            "filtered_candidates": len(candidates),
            "inside_fov": len(inside_fov),
            "fov_radius": float(radius),
            "fov_ratio": float(fov_ratio),
            "min_confidence": float(min_confidence),
            "class_filter": str(class_filter),
            "sticky_bias": float(sticky_bias),
            "lock_enabled": bool(lock_enabled),
            "selected": self._candidate_debug([best], context)[0] if best is not None else None,
            "candidates": self._candidate_debug(inside_fov, context),
        }
        return TargetSelection(
            target=best,
            state="locked" if selected_locked else "acquire" if previous_key != self._target_key(best) else "fresh",
            reason="按距离选目标，锁定偏好和类别优先级参与排序",
            candidates=len(candidates),
            inside_fov=len(inside_fov),
            locked=selected_locked,
            priority_rank=self._priority_rank(best, priority),
            distance_px=self._distance(best, center_x, center_y),
        )

    def _lost_or_clear(
        self,
        grace: int,
        reason: str,
        candidates: int,
        inside_fov: int,
    ) -> TargetSelection:
        if self._locked is None:
            return TargetSelection(
                None,
                "no_target",
                reason,
                candidates=candidates,
                inside_fov=inside_fov,
                lost_count=0,
            )
        if self._lost_count <= max(0, grace):
            return TargetSelection(
                None,
                "lost",
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
            "reacquire",
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
        return None

    def _selection_distance(
        self,
        target: Target,
        locked: Target | None,
        sticky_bias: float,
        center_x: float,
        center_y: float,
    ) -> float:
        distance = self._distance(target, center_x, center_y)
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

    def _candidate_debug(self, candidates: list[Target], context: FrameContext) -> list[dict]:
        center_x = context.width / 2
        center_y = context.height / 2
        return [
            {
                "cls": int(getattr(item, "cls", -1)),
                "score": float(getattr(item, "score", 0.0)),
                "cx": float(getattr(item, "cx", 0.0)),
                "cy": float(getattr(item, "cy", 0.0)),
                "distance_px": self._distance(item, center_x, center_y),
            }
            for item in candidates[:12]
        ]
