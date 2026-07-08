from __future__ import annotations

import importlib
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from novasight.contracts import Detection, DetectionBatch, DetectionCoordinateSpace


TimestampConverter = Callable[[int, int], tuple[int, str] | int]


@dataclass(frozen=True)
class DetectionProbeConfig:
    slot: DetectionSlot
    classes: Sequence[str] = field(default_factory=tuple)
    coordinate_space: DetectionCoordinateSpace = "roi"
    timestamp_converter: TimestampConverter | None = None


class DetectionSlot:
    """Single-slot handoff for the latest DeepStream object-meta DetectionBatch."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._batch: DetectionBatch | None = None
        self._version = 0

    def put(self, batch: DetectionBatch) -> None:
        with self._condition:
            self._batch = batch
            self._version += 1
            self._condition.notify_all()

    def get(
        self,
        *,
        after_version: int | None = None,
        timeout_s: float | None = None,
    ) -> DetectionBatch | None:
        deadline = None if timeout_s is None else time.monotonic() + max(0.0, float(timeout_s))
        with self._condition:
            while self._batch is None or (
                after_version is not None and self._version <= int(after_version)
            ):
                if timeout_s == 0:
                    return None
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
            return self._batch

    @property
    def version(self) -> int:
        with self._condition:
            return self._version


def detection_probe(_pad: Any, info: Any, user_data: Any) -> Any:
    """GStreamer pad probe that publishes DeepStream object metadata to a DetectionSlot."""

    Gst = _import_gst()
    try:
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK
        pyds = importlib.import_module("pyds")
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        if batch_meta is None:
            return Gst.PadProbeReturn.OK
        config = _coerce_probe_config(user_data)
        observed_ns = time.monotonic_ns()
        for batch in iter_detection_batches_from_batch_meta(
            batch_meta,
            pyds=pyds,
            observed_ns=observed_ns,
            classes=config.classes,
            coordinate_space=config.coordinate_space,
            timestamp_converter=config.timestamp_converter,
        ):
            config.slot.put(batch)
    except Exception:
        return Gst.PadProbeReturn.OK
    return Gst.PadProbeReturn.OK


def iter_detection_batches_from_batch_meta(
    batch_meta: Any,
    *,
    pyds: Any,
    observed_ns: int,
    classes: Sequence[str] = (),
    coordinate_space: DetectionCoordinateSpace = "roi",
    timestamp_converter: TimestampConverter | None = None,
) -> Iterator[DetectionBatch]:
    frame_list = getattr(batch_meta, "frame_meta_list", None)
    while frame_list is not None:
        frame_meta = pyds.NvDsFrameMeta.cast(frame_list.data)
        detections = list(_iter_frame_detections(frame_meta, pyds=pyds))
        capture_ts_ns, timestamp_source = _capture_timestamp(
            frame_meta,
            observed_ns=int(observed_ns),
            timestamp_converter=timestamp_converter,
        )
        frame_id = _frame_id(frame_meta)
        yield DetectionBatch(
            frame_id=frame_id,
            capture_ts_ns=capture_ts_ns,
            inference_start_ts_ns=capture_ts_ns,
            inference_end_ts_ns=int(observed_ns),
            detections=detections,
            classes=_classes_for_detections(classes, detections),
            coordinate_space=coordinate_space,
            metadata={
                "source": "deepstream-object-meta",
                "timestamp_source": timestamp_source,
                "frame_num": frame_id,
                "batch_id": _optional_int(frame_meta, "batch_id"),
                "source_id": _optional_int(frame_meta, "source_id"),
                "buf_pts": _optional_int(frame_meta, "buf_pts"),
                "ntp_timestamp": _optional_int(frame_meta, "ntp_timestamp"),
                "detection_count": len(detections),
            },
        )
        frame_list = frame_list.next


def _iter_frame_detections(frame_meta: Any, *, pyds: Any) -> Iterator[Detection]:
    obj_list = getattr(frame_meta, "obj_meta_list", None)
    while obj_list is not None:
        obj_meta = pyds.NvDsObjectMeta.cast(obj_list.data)
        rect = getattr(obj_meta, "rect_params", None)
        if rect is not None:
            left = float(getattr(rect, "left", 0.0) or 0.0)
            top = float(getattr(rect, "top", 0.0) or 0.0)
            width = float(getattr(rect, "width", 0.0) or 0.0)
            height = float(getattr(rect, "height", 0.0) or 0.0)
            yield Detection(
                cls=int(getattr(obj_meta, "class_id", 0) or 0),
                score=float(getattr(obj_meta, "confidence", 0.0) or 0.0),
                x=left,
                y=top,
                w=width,
                h=height,
            )
        obj_list = obj_list.next


def _capture_timestamp(
    frame_meta: Any,
    *,
    observed_ns: int,
    timestamp_converter: TimestampConverter | None,
) -> tuple[int, str]:
    raw_pts = int(getattr(frame_meta, "buf_pts", 0) or 0)
    if timestamp_converter is not None:
        converted = timestamp_converter(raw_pts, int(observed_ns))
        if isinstance(converted, tuple):
            timestamp_ns, source = converted
            return _positive_timestamp(timestamp_ns, observed_ns), str(source)
        return _positive_timestamp(converted, observed_ns), "converted_buf_pts"
    if raw_pts > 0:
        return raw_pts, "buf_pts"
    return int(observed_ns), "observed_probe_time_missing_pts"


def _coerce_probe_config(user_data: Any) -> DetectionProbeConfig:
    if isinstance(user_data, DetectionProbeConfig):
        return user_data
    if isinstance(user_data, DetectionSlot):
        return DetectionProbeConfig(slot=user_data)
    raise TypeError("detection_probe user_data must be DetectionProbeConfig or DetectionSlot")


def _classes_for_detections(classes: Sequence[str], detections: Sequence[Detection]) -> list[str]:
    result = list(classes)
    if not detections:
        return result
    class_count = max(int(item.cls) for item in detections) + 1
    if len(result) < class_count:
        result.extend(str(index) for index in range(len(result), class_count))
    return result


def _frame_id(frame_meta: Any) -> int:
    return max(0, int(getattr(frame_meta, "frame_num", 0) or 0))


def _optional_int(obj: Any, attr: str) -> int | None:
    value = getattr(obj, attr, None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _positive_timestamp(timestamp_ns: int, observed_ns: int) -> int:
    value = int(timestamp_ns or 0)
    if value > 0:
        return value
    return int(observed_ns)


def _import_gst() -> Any:
    gi = importlib.import_module("gi")
    gi.require_version("Gst", "1.0")
    Gst = importlib.import_module("gi.repository.Gst")
    init = getattr(Gst, "init", None)
    is_initialized = getattr(Gst, "is_initialized", None)
    if callable(init) and (not callable(is_initialized) or not is_initialized()):
        init(None)
    return Gst


__all__ = [
    "DetectionProbeConfig",
    "DetectionSlot",
    "TimestampConverter",
    "detection_probe",
    "iter_detection_batches_from_batch_meta",
]
