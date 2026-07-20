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
    metadata = getattr(detection_batch, "metadata", {}) or {}
    return FrameContext(
        frame_id=int(detection_batch.frame_id),
        width=int(width),
        height=int(height),
        generation=int(
            detection_batch.frame_id
            if detection_batch.generation is None
            else detection_batch.generation
        ),
        detections=list(detection_batch.detections),
        tracks=detection_batch_tracks(detection_batch),
        classes=list(detection_batch.classes),
        capture_ts_ns=int(detection_batch.capture_ts_ns),
        dequeue_ts_ns=_metadata_int(
            metadata,
            "dequeue_ts_ns",
            "dequeue_timestamp_ns",
        ),
        decode_ts_ns=_metadata_int(
            metadata,
            "decode_ts_ns",
            "decode_timestamp_ns",
        ),
        roi_ts_ns=_metadata_int(
            metadata,
            "roi_ts_ns",
            "roi_timestamp_ns",
        ),
        inference_start_ts_ns=int(detection_batch.inference_start_ts_ns),
        inference_end_ts_ns=int(detection_batch.inference_end_ts_ns),
        postprocess_ts_ns=_metadata_int(
            metadata,
            "postprocess_ts_ns",
            "postprocess_timestamp_ns",
        )
        or int(detection_batch.inference_end_ts_ns),
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
            velocity_px_s=_velocity(value.get("velocity_px_s")),
            missed_frames=int(value.get("missed_frames", 0) or 0),
            last_seen_ns=int(value.get("last_seen_ns", 0) or 0),
            state_ts_ns=int(value.get("state_ts_ns", 0) or 0),
            state_valid=(
                bool(value["state_valid"])
                if value.get("state_valid") is not None
                else None
            ),
            prediction_confidence=float(value.get("prediction_confidence", 1.0) or 0.0),
            identity_confidence=float(value.get("identity_confidence", 1.0) or 0.0),
            track_rebuilt=bool(value.get("track_rebuilt", False)),
            is_predicted=bool(value.get("is_predicted", False)),
            is_stale=bool(value.get("is_stale", False)),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _velocity(value: Any) -> tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    if isinstance(value, Mapping):
        return float(value.get("x", 0.0) or 0.0), float(value.get("y", 0.0) or 0.0)
    return 0.0, 0.0


def _metadata_int(metadata: object, *keys: str) -> int | None:
    if not isinstance(metadata, Mapping):
        return None
    for key in keys:
        value = metadata.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


__all__ = [
    "detection_batch_to_frame_context",
    "detection_batch_tracks",
]
