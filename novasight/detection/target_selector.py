from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from novasight.contracts import FrameContext, Track
from novasight.runtime.candidates import (
    CandidateFilter as RuntimeCandidateFilter,
    CandidateFilterConfig,
    QualityScoreConfig,
    QualityScorer as RuntimeQualityScorer,
    RatioCheckConfig,
    SelectionFovConfig,
    parse_allowed_class_ids,
)


@dataclass(frozen=True)
class TargetSelectorConfig:
    frame_width: int
    frame_height: int
    min_confidence: float = 0.0
    class_filter: str = "all"
    switch_cost: float = 0.08
    aim_ratio: float = 50.0
    fov: SelectionFovConfig = field(default_factory=SelectionFovConfig)
    ratio: RatioCheckConfig = field(default_factory=RatioCheckConfig)
    quality: QualityScoreConfig = field(default_factory=QualityScoreConfig)


class CandidateFilter:
    def __init__(self, config: TargetSelectorConfig) -> None:
        self.config = config

    def filter(self, tracks: list[Track]) -> list[Track]:
        context = _context_for_tracks(tracks, self.config)
        result = RuntimeCandidateFilter(
            CandidateFilterConfig(
                allowed_class_ids=parse_allowed_class_ids(self.config.class_filter),
                min_confidence=max(0.0, float(self.config.min_confidence)),
                selection_fov=self.config.fov,
                ratio_check=self.config.ratio,
            )
        ).apply(context, tracks, aim_ratio=self.config.aim_ratio)
        return [item.detection for item in result.candidates if isinstance(item.detection, Track)]


class QualityScorer:
    def __init__(self, config: TargetSelectorConfig) -> None:
        self.config = config
        self._scorer = RuntimeQualityScorer(config.quality)

    def score(self, track: Track, current_time: int) -> float:
        context = _context_for_tracks([track], self.config, now_ns=current_time)
        return float(self._scorer.score(track, context=context).quality_score)


class TargetSelector:
    def __init__(self, config: TargetSelectorConfig) -> None:
        self.config = config
        self.filter = CandidateFilter(config)
        self.scorer = QualityScorer(config)
        self.current_target: Track | None = None
        self.current_score = 0.0
        self.switch_cost = max(0.0, float(config.switch_cost))

    def select(self, tracks: list[Track], now_ns: int) -> Track | None:
        candidates = self.filter.filter(tracks)
        if not candidates:
            return self.handle_lost()
        scored = sorted(
            ((self.scorer.score(track, now_ns), track) for track in candidates),
            key=lambda item: item[0],
            reverse=True,
        )
        best_score, best_track = scored[0]
        current = self._current_candidate(candidates)
        if current is not None and int(current.track_id) != int(best_track.track_id):
            current_score = self.scorer.score(current, now_ns)
            if best_score < current_score + self.switch_cost:
                best_track = current
                best_score = current_score
        self.current_target = best_track
        self.current_score = float(best_score)
        return best_track

    def handle_lost(self) -> Track | None:
        return self.current_target

    def _current_candidate(self, candidates: Iterable[Track]) -> Track | None:
        if self.current_target is None:
            return None
        current_id = int(self.current_target.track_id)
        for track in candidates:
            if int(track.track_id) == current_id:
                return track
        return None


def _context_for_tracks(
    tracks: list[Track],
    config: TargetSelectorConfig,
    *,
    now_ns: int | None = None,
) -> FrameContext:
    return FrameContext(
        frame_id=0,
        width=max(1, int(config.frame_width)),
        height=max(1, int(config.frame_height)),
        tracks=list(tracks),
        capture_ts_ns=int(now_ns) if now_ns is not None else None,
    )
