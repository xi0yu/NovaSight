from __future__ import annotations

import importlib
import threading
import time
from dataclasses import dataclass
from typing import Any

from novasight.contracts import DetectionBatch
from novasight.model_registry.manifest import ModelManifest

from .pipeline_builder import DeepStreamPipelineConfig, build_deepstream_pipeline
from .tensor_meta import output_tensor_to_detection_batch


@dataclass(frozen=True)
class DeepStreamDependencyStatus:
    available: bool
    reason: str = ""
    detail: str = ""


class DeepStreamDetectionBackend:
    """Experimental DeepStream backend that publishes NovaSight DetectionBatch objects.

    The class is intentionally isolated from the legacy runtime. On non-Jetson
    development machines it reports missing GStreamer/pyds dependencies instead
    of silently falling back to CPU capture.
    """

    backend_id = "deepstream"

    def __init__(
        self,
        *,
        pipeline_config: DeepStreamPipelineConfig,
        manifest: ModelManifest,
        roi_width: int,
        roi_height: int,
    ) -> None:
        self.pipeline_config = pipeline_config
        self.manifest = manifest
        self.roi_width = int(roi_width)
        self.roi_height = int(roi_height)
        self.pipeline_description = build_deepstream_pipeline(pipeline_config)
        self._lock = threading.RLock()
        self._pipeline: Any | None = None
        self._running = False
        self._last_result: DetectionBatch | None = None
        self._last_error = ""
        self._published_batches = 0
        self._started_at_ns = 0

    def dependency_status(self) -> DeepStreamDependencyStatus:
        try:
            gi = importlib.import_module("gi")
            gi.require_version("Gst", "1.0")
            importlib.import_module("gi.repository.Gst")
        except Exception as exc:
            return DeepStreamDependencyStatus(
                available=False,
                reason="gstreamer-python-unavailable",
                detail=str(exc),
            )
        try:
            importlib.import_module("pyds")
        except Exception as exc:
            return DeepStreamDependencyStatus(
                available=False,
                reason="pyds-unavailable",
                detail=str(exc),
            )
        return DeepStreamDependencyStatus(available=True)

    def start(self) -> None:
        dependency = self.dependency_status()
        if not dependency.available:
            self._last_error = dependency.detail or dependency.reason
            raise RuntimeError(f"DeepStream backend unavailable: {dependency.reason}: {dependency.detail}")
        Gst = importlib.import_module("gi.repository.Gst")
        Gst.init(None)
        pipeline = Gst.parse_launch(self.pipeline_description)
        self._attach_tensor_probe(pipeline)
        result = pipeline.set_state(Gst.State.PLAYING)
        if result == Gst.StateChangeReturn.FAILURE:
            self._last_error = "failed to set DeepStream pipeline to PLAYING"
            raise RuntimeError(self._last_error)
        with self._lock:
            self._pipeline = pipeline
            self._running = True
            self._started_at_ns = time.monotonic_ns()

    def stop(self) -> None:
        with self._lock:
            pipeline = self._pipeline
            self._pipeline = None
            self._running = False
        if pipeline is None:
            return
        try:
            Gst = importlib.import_module("gi.repository.Gst")
            pipeline.set_state(Gst.State.NULL)
        except Exception as exc:
            self._last_error = str(exc)

    def latest_result(self, after_frame_id: int | None = None) -> DetectionBatch | None:
        with self._lock:
            result = self._last_result
        if result is None:
            return None
        if after_frame_id is not None and result.frame_id <= int(after_frame_id):
            return None
        return result

    def status(self) -> dict[str, Any]:
        dependency = self.dependency_status()
        with self._lock:
            running = self._running
            last_result = self._last_result
            published = self._published_batches
            started_at_ns = self._started_at_ns
        uptime_ms = 0.0
        if started_at_ns:
            uptime_ms = max(0.0, (time.monotonic_ns() - started_at_ns) / 1e6)
        return {
            "selected": self.backend_id,
            "available": dependency.available,
            "reason": dependency.reason,
            "detail": dependency.detail,
            "running": running,
            "published_batches": published,
            "last_frame_id": last_result.frame_id if last_result else 0,
            "last_error": self._last_error,
            "uptime_ms": uptime_ms,
            "pipeline": self.pipeline_description,
        }

    def publish_output_tensor(
        self,
        output: Any,
        *,
        frame_id: int,
        capture_ts_ns: int,
        inference_start_ts_ns: int,
        inference_end_ts_ns: int,
    ) -> DetectionBatch:
        batch = output_tensor_to_detection_batch(
            output,
            manifest=self.manifest,
            frame_id=frame_id,
            capture_ts_ns=capture_ts_ns,
            inference_start_ts_ns=inference_start_ts_ns,
            inference_end_ts_ns=inference_end_ts_ns,
            roi_width=self.roi_width,
            roi_height=self.roi_height,
        )
        with self._lock:
            self._last_result = batch
            self._published_batches += 1
            self._last_error = ""
        return batch

    def _attach_tensor_probe(self, pipeline: Any) -> None:
        nvinfer = pipeline.get_by_name("primary-infer")
        if nvinfer is None:
            raise RuntimeError("DeepStream pipeline missing primary-infer element")
        src_pad = nvinfer.get_static_pad("src")
        if src_pad is None:
            raise RuntimeError("DeepStream primary-infer element has no src pad")
        Gst = importlib.import_module("gi.repository.Gst")
        src_pad.add_probe(Gst.PadProbeType.BUFFER, self._tensor_probe)

    def _tensor_probe(self, _pad: Any, info: Any) -> Any:
        Gst = importlib.import_module("gi.repository.Gst")
        try:
            buffer = info.get_buffer()
            if buffer is None:
                return Gst.PadProbeReturn.OK
            for frame_id, capture_ts_ns, output in self._iter_output_tensors(buffer):
                done_ns = time.monotonic_ns()
                self.publish_output_tensor(
                    output,
                    frame_id=frame_id,
                    capture_ts_ns=capture_ts_ns,
                    inference_start_ts_ns=done_ns,
                    inference_end_ts_ns=done_ns,
                )
        except Exception as exc:
            self._last_error = str(exc)
        return Gst.PadProbeReturn.OK

    def _iter_output_tensors(self, buffer: Any) -> Any:
        """Yield output tensors from NvDsInferTensorMeta.

        This method is kept small and defensive because it can only be fully
        exercised on Jetson with DeepStream's pyds module installed.
        """
        pyds = importlib.import_module("pyds")
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        if batch_meta is None:
            return
        frame_list = batch_meta.frame_meta_list
        while frame_list is not None:
            frame_meta = pyds.NvDsFrameMeta.cast(frame_list.data)
            frame_id = int(getattr(frame_meta, "frame_num", 0))
            pts = int(getattr(frame_meta, "buf_pts", 0) or getattr(buffer, "pts", 0) or time.monotonic_ns())
            user_meta_list = frame_meta.frame_user_meta_list
            while user_meta_list is not None:
                user_meta = pyds.NvDsUserMeta.cast(user_meta_list.data)
                meta_type = getattr(user_meta, "base_meta", None)
                meta_type_value = getattr(meta_type, "meta_type", None)
                tensor_meta_type = getattr(getattr(pyds, "NvDsMetaType", object), "NVDSINFER_TENSOR_OUTPUT_META", None)
                if meta_type_value == tensor_meta_type:
                    tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                    output = _tensor_meta_first_output_to_numpy(pyds, tensor_meta)
                    if output is not None:
                        yield frame_id, pts, output
                user_meta_list = user_meta_list.next
            frame_list = frame_list.next


def _tensor_meta_first_output_to_numpy(pyds: Any, tensor_meta: Any) -> Any:
    import ctypes

    import numpy as np

    layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
    dims = getattr(layer, "inferDims", None)
    shape = _layer_shape(dims)
    if not shape:
        return None
    ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
    array = np.ctypeslib.as_array(ptr, shape=tuple(shape))
    return np.array(array, copy=True)


def _layer_shape(dims: Any) -> list[int]:
    if dims is None:
        return []
    if hasattr(dims, "d") and hasattr(dims, "numDims"):
        values = [int(dims.d[index]) for index in range(int(dims.numDims))]
        return values
    if hasattr(dims, "numElements"):
        return [int(dims.numElements)]
    return []
