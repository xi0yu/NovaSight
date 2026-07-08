from __future__ import annotations

from novasight.contracts import BBox, Track
from novasight.runtime.kalman import (
    KalmanConfig,
    KalmanEstimator as RuntimeKalmanEstimator,
)


class PredictedTrack(Track):
    def __init__(
        self,
        *,
        track_id: int,
        cls: int,
        score: float,
        box: BBox,
        velocity_px_s: tuple[float, float],
        quality_score: float,
        missed_frames: int,
        last_seen_ns: int,
        is_predicted: bool = True,
        is_stale: bool = False,
    ) -> None:
        Track.__init__(
            self,
            track_id=track_id,
            cls=cls,
            score=score,
            box=box,
            velocity_px_s=velocity_px_s,
            quality_score=quality_score,
            missed_frames=missed_frames,
            last_seen_ns=last_seen_ns,
            is_predicted=is_predicted,
            is_stale=is_stale,
        )


class KalmanEstimator:
    def __init__(self, *, ttl_ns: int = 50_000_000, config: KalmanConfig | None = None) -> None:
        self.ttl_ns = max(1, int(ttl_ns))
        self.config = config or KalmanConfig(max_predict_missing_ms=self.ttl_ns / 1e6)
        self.kf: RuntimeKalmanEstimator | None = None
        self.last_update_ns = 0
        self._last_track: Track | None = None

    def update(self, track: Track, now_ns: int) -> None:
        now_ns = int(now_ns)
        if self.kf is None or self._last_track is None or int(track.track_id) != int(self._last_track.track_id):
            self.kf = RuntimeKalmanEstimator(
                track_id=int(track.track_id),
                x=float(track.cx),
                y=float(track.cy),
                ts_ns=now_ns,
                config=self.config,
            )
        else:
            self.kf.update(
                measurement_x=float(track.cx),
                measurement_y=float(track.cy),
                ts_ns=now_ns,
                identity_confidence=1.0,
            )
        self.last_update_ns = now_ns
        self._last_track = track

    def predict(self, now_ns: int) -> Track | None:
        if self.kf is None or self._last_track is None or self.last_update_ns <= 0:
            return None
        age_ns = int(now_ns) - int(self.last_update_ns)
        if age_ns < 0 or age_ns > self.ttl_ns:
            return None
        estimate = self.kf.predict_only(ts_ns=int(now_ns), identity_confidence=1.0)
        if not estimate.valid:
            return None
        previous = self._last_track
        return PredictedTrack(
            track_id=int(previous.track_id),
            cls=int(previous.cls),
            score=float(previous.score),
            box=BBox.from_cxcywh(
                estimate.x,
                estimate.y,
                max(1.0, float(previous.w)),
                max(1.0, float(previous.h)),
            ),
            velocity_px_s=(float(estimate.vx), float(estimate.vy)),
            quality_score=float(estimate.prediction_confidence),
            missed_frames=int(estimate.prediction_steps),
            last_seen_ns=int(self.last_update_ns),
            is_predicted=True,
            is_stale=False,
        )


__all__ = [
    "KalmanEstimator",
    "PredictedTrack",
]
