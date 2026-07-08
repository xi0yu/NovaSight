from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from novasight.contracts import DetectionBatch, FrameContext, Track


def detection_batch_tracks(detection_batch: DetectionBatch) -> list[Track]:
    """Restore DeepStream nvtracker objects stored in DetectionBatch metadata."""

    metadata = getattr(detection_batch, "metadata", {}) or {}
    raw_tracks = metadata.get("tracks")
    if not isinstance(raw_tracks, list):
        return []
    tracks: list[Track] = []
    for item in raw_tracks:
        track = _track_from_mapping(item)
        if track is not None:
            tracks.append(track)
    return tracks


def detection_batch_to_frame_context(
    detection_batch: DetectionBatch,
    *,
    width: int,
    height: int,
) -> FrameContext:
    return FrameContext(
        frame_id=int(detection_batch.frame_id),
        width=int(width),
        height=int(height),
        detections=list(detection_batch.detections),
        tracks=detection_batch_tracks(detection_batch),
        classes=list(detection_batch.classes),
        capture_ts_ns=int(detection_batch.capture_ts_ns),
        inference_start_ts_ns=int(detection_batch.inference_start_ts_ns),
        inference_end_ts_ns=int(detection_batch.inference_end_ts_ns),
        postprocess_ts_ns=int(detection_batch.inference_end_ts_ns),
    )


def _track_from_mapping(value: Any) -> Track | None:
    if not isinstance(value, Mapping):
        return None
    try:
        return Track(
            track_id=int(value["track_id"]),
            cls=int(value.get("cls", value.get("class_id", 0))),
            score=float(value.get("score", value.get("confidence", 0.0))),
            x1=float(value["x1"]),
            y1=float(value["y1"]),
            x2=float(value["x2"]),
            y2=float(value["y2"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


__all__ = [
    "detection_batch_to_frame_context",
    "detection_batch_tracks",
]
