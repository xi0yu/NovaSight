from __future__ import annotations

import ctypes
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import InferenceDetection, InferenceResult
from .input import (
    PreparedTensorInput,
    TensorInputShape,
    normalize_tensor_dtype,
    prepare_tensor_input,
)
from .geometry import map_model_detections_to_roi_frame, preprocess_debug
from .postprocess.yolo import decode_nx6_detections
from .preprocess import DeviceTensor, GpuResourcePreprocessor, prepare_tensor


logger = logging.getLogger("novasight.inference.tensorrt")


def _decoder_candidate_columns(shape: tuple[int, ...]) -> int | None:
    squeezed = tuple(int(item) for item in shape if int(item) != 1)
    if len(squeezed) != 2:
        return None
    rows, columns = squeezed
    transposed_from_channel_first = 4 < rows < columns
    candidate_columns = rows if transposed_from_channel_first else columns
    return candidate_columns if candidate_columns >= 5 else None


def _select_detection_output_name(
    output_names: list[str],
    output_shapes: dict[str, tuple[int, ...]],
    *,
    class_count: int,
) -> tuple[str | None, int | None]:
    expected_columns = {4 + max(0, class_count), 5 + max(0, class_count), 6}
    candidates: list[tuple[int, int, str, int]] = []
    for index, name in enumerate(output_names):
        columns = _decoder_candidate_columns(output_shapes.get(name, ()))
        if columns is None:
            continue
        rank = 0 if columns in expected_columns else 1
        candidates.append((rank, index, name, columns))
    if not candidates:
        return None, None
    _rank, _index, name, columns = min(candidates)
    return name, columns


class TensorRtInferenceEngine:
    engine_id = "tensorrt"

    def __init__(
        self,
        *,
        confidence_threshold: float = 0.25,
        nms_threshold: float = 0.45,
        gpu_preprocessor: GpuResourcePreprocessor | None = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self._gpu_preprocessor = gpu_preprocessor
        self._reason = ""
        self._available = False
        self._loaded = False
        self._warmed = False
        self._classes: list[str] = []
        self._input_shape: TensorInputShape | None = None
        self._last_input: PreparedTensorInput | None = None
        self._trt: Any | None = None
        self._cudart: Any | None = None
        self._engine: Any | None = None
        self._context: Any | None = None
        self._stream: Any | None = None
        self._input_name = ""
        self._output_name = ""
        self._device_input: int | None = None
        self._host_input: Any | None = None
        self._host_allocations: list[_HostAllocation] = []
        self._device_outputs: dict[str, int] = {}
        self._host_outputs: dict[str, Any] = {}
        self._output_shapes: dict[str, tuple[int, ...]] = {}
        self._output_dtypes: dict[str, str] = {}
        self._output_shape: tuple[int, ...] = ()
        self._output_dtype = ""
        self._output_candidate_columns: int | None = None
        self._input_dtype = "float32"
        self._last_failure_logged = ""
        self._engine_input_shape: tuple[int, ...] = ()
        self._input_profile_shapes: dict[str, tuple[int, ...]] = {}
        self._input_shape_source = ""
        self._io_tensors: list[dict[str, Any]] = []
        self._last_slow_log_ns = 0
        self._last_preprocess_backend = ""
        self._last_preprocess_reason = ""
        self._last_preprocess_timings: dict[str, float] = {}

    def available(self) -> bool:
        try:
            import tensorrt
            try:
                from cuda.bindings import runtime as cudart
            except ImportError:
                from cuda import cudart  # type: ignore
        except Exception as exc:
            self._reason = f"TensorRT unavailable: {exc}"
            self._available = False
            return False
        self._trt = tensorrt
        self._cudart = cudart
        self._reason = ""
        self._available = True
        return True

    def last_reason(self) -> str:
        return self._reason

    def status(self) -> dict:
        return {
            "selected": self.engine_id,
            "available": self._available,
            "loaded": self._loaded,
            "warmed": self._warmed,
            "supports_execution": True,
            "device": "cuda",
            "require_gpu": True,
            "allow_cpu_fallback": False,
            "reason": self._reason,
            "input_name": self._input_name,
            "input_shape": str(self._input_shape) if self._input_shape is not None else "",
            "input_dtype": self._input_dtype,
            "output_shape": "x".join(str(item) for item in self._output_shape),
            "output_name": self._output_name,
            "output_dtype": self._output_dtype,
            "output_candidate_columns": self._output_candidate_columns,
            "engine_input_shape": "x".join(str(item) for item in self._engine_input_shape),
            "input_shape_source": self._input_shape_source,
            "host_input_pinned": self._host_input_pinned(),
            "host_output_pinned": self._host_output_pinned(),
            "input_profile": {
                name: list(shape)
                for name, shape in self._input_profile_shapes.items()
            },
            "outputs": {
                name: {
                    "shape": list(self._output_shapes.get(name, ())),
                    "dtype": self._output_dtypes.get(name, ""),
                }
                for name in self._output_shapes
            },
            "io_tensors": [dict(tensor) for tensor in self._io_tensors],
            "confidence_threshold": self.confidence_threshold,
            "nms_threshold": self.nms_threshold,
            "last_input_mode": self._last_input.mode if self._last_input is not None else "",
            "last_input_frame_id": (
                self._last_input.frame_id if self._last_input is not None else 0
            ),
            "last_input_capture_ts_ns": (
                self._last_input.capture_ts_ns if self._last_input is not None else 0
            ),
            "last_input_resource_kind": (
                self._last_input.resource_kind if self._last_input is not None else ""
            ),
            "last_input_resource_memory": (
                self._last_input.resource_memory if self._last_input is not None else ""
            ),
            "last_input_resource_source": (
                self._last_input.resource_source if self._last_input is not None else ""
            ),
            "last_input_resource_size": (
                f"{self._last_input.resource_width}x{self._last_input.resource_height}"
                if self._last_input is not None
                and self._last_input.resource_width > 0
                and self._last_input.resource_height > 0
                else ""
            ),
            "last_input_resource_format": (
                self._last_input.resource_pixel_format if self._last_input is not None else ""
            ),
            "last_input_dmabuf_fd": (
                self._last_input.dmabuf_fd if self._last_input is not None else None
            ),
            "last_input_gst_buffer_ptr": (
                self._last_input.gst_buffer_ptr if self._last_input is not None else None
            ),
            "last_input_needs_resize": (
                self._last_input.needs_resize if self._last_input is not None else False
            ),
            "last_preprocess_backend": self._last_preprocess_backend,
            "last_preprocess_reason": self._last_preprocess_reason,
            "last_preprocess_timings": dict(self._last_preprocess_timings),
            "gpu_preprocessor": self._gpu_preprocessor_status(),
        }

    def set_gpu_preprocessor(self, gpu_preprocessor: GpuResourcePreprocessor | None) -> None:
        self._gpu_preprocessor = gpu_preprocessor

    def _gpu_preprocessor_status(self) -> dict[str, Any]:
        if self._gpu_preprocessor is None:
            return {
                "selected": "",
                "available": False,
                "reason": "disabled",
            }
        status = getattr(self._gpu_preprocessor, "status", None)
        if callable(status):
            return dict(status())
        return {
            "selected": type(self._gpu_preprocessor).__name__,
            "available": True,
            "reason": "",
        }

    def load(self, artifact_path: Path, classes: list[str], input_shape: str) -> None:
        if not artifact_path.suffix == ".engine":
            raise ValueError(f"TensorRT artifact must be .engine: {artifact_path}")
        if not classes:
            raise ValueError("TensorRT classes must not be empty")
        if not self.available():
            raise RuntimeError(self._reason)
        self.close()
        self._classes = list(classes)
        self._last_input = None
        self._last_preprocess_backend = ""
        self._last_preprocess_reason = ""
        self._last_preprocess_timings = {}
        self._load_engine(artifact_path, requested_shape=None)
        self._loaded = True
        self._warmup()

    def infer(self, frame: Any) -> InferenceResult:
        if not self._loaded:
            return InferenceResult(available=False, reason="TensorRT engine not loaded")
        if self._input_shape is None or self._context is None or self._cudart is None:
            return InferenceResult(available=False, reason="TensorRT input shape not loaded")
        timings: dict[str, float] = {}
        total_start_ns = time.monotonic_ns()
        try:
            prepare_start_ns = time.monotonic_ns()
            self._last_input = prepare_tensor_input(frame, self._input_shape)
            tensor_start_ns = time.monotonic_ns()
            preprocess_result = prepare_tensor(
                self._last_input,
                self._input_shape,
                gpu_preprocessor=self._gpu_preprocessor,
            )
            self._last_preprocess_backend = preprocess_result.backend
            self._last_preprocess_reason = preprocess_result.reason
            self._last_preprocess_timings = dict(preprocess_result.timings or {})
            tensor = preprocess_result.tensor
            execute_start_ns = time.monotonic_ns()
            detections, decode_debug = self._execute(tensor)
            scale_start_ns = time.monotonic_ns()
            detections = map_model_detections_to_roi_frame(
                detections,
                prepared=self._last_input,
                shape=self._input_shape,
                preprocess_result=preprocess_result,
            )
            preprocess_debug_payload = preprocess_debug(
                self._last_input,
                self._input_shape,
                preprocess_result=preprocess_result,
            )
            done_ns = time.monotonic_ns()
            timings.update(
                {
                    "prepare_input_ms": _elapsed_ms(prepare_start_ns, tensor_start_ns),
                    "numpy_tensor_ms": _elapsed_ms(tensor_start_ns, execute_start_ns),
                    "execute_total_ms": _elapsed_ms(execute_start_ns, scale_start_ns),
                    "scale_ms": _elapsed_ms(scale_start_ns, done_ns),
                    "total_ms": _elapsed_ms(total_start_ns, done_ns),
                }
            )
            for key, value in self._last_preprocess_timings.items():
                timings[f"native_preprocess_{key}"] = float(value)
            self._log_slow_inference(timings, decode_debug)
        except ValueError as exc:
            self._last_preprocess_backend = "unavailable"
            self._last_preprocess_reason = getattr(exc, "reason", "")
            self._last_preprocess_timings = {}
            self._log_failure_once(f"input rejected: {exc}")
            return InferenceResult(available=False, reason=str(exc))
        except Exception as exc:
            self._last_preprocess_backend = "unavailable"
            self._last_preprocess_reason = getattr(exc, "reason", "")
            self._last_preprocess_timings = {}
            self._log_failure_once(str(exc), with_trace=True)
            return InferenceResult(available=False, reason=str(exc), classes=self._classes)
        return InferenceResult(
            available=True,
            detections=detections,
            classes=self._classes,
            debug={
                "engine": self.engine_id,
                "input_name": self._input_name,
                "output_name": self._output_name,
                "output_shape": list(self._output_shape),
                "output_dtype": self._output_dtype,
                "decoded_detections": len(detections),
                    "preprocess": preprocess_debug_payload,
                "decode": decode_debug,
                "timings": timings,
            },
        )

    def _warmup(self) -> None:
        if self._input_shape is None:
            self._warmed = False
            return
        try:
            import numpy as np

            logger.info("TensorRT warmup starting input=%s output=%s", self._input_shape, self._output_name)
            dummy = np.zeros(
                (self._input_shape.height, self._input_shape.width, 3),
                dtype=np.uint8,
            )
            result = self.infer(type("_Frame", (), {
                "image": dummy,
                "width": self._input_shape.width,
                "height": self._input_shape.height,
                "pixel_format": "BGR",
            })())
            self._warmed = result.available
            if not result.available:
                logger.warning("TensorRT warmup failed: %s", result.reason)
            else:
                logger.info("TensorRT warmup complete input=%s output=%s", self._input_shape, self._output_name)
        except Exception as exc:
            self._warmed = False
            logger.warning("TensorRT warmup failed: %s", exc)

    def _load_engine(
        self,
        artifact_path: Path,
        *,
        requested_shape: tuple[int, int, int, int] | None,
    ) -> None:
        import numpy as np

        trt = self._trt
        cudart = self._cudart
        if trt is None or cudart is None:
            raise RuntimeError("TensorRT runtime is not available")
        logger_trt = trt.Logger(trt.Logger.WARNING)
        with artifact_path.open("rb") as handle, trt.Runtime(logger_trt) as runtime:
            engine = runtime.deserialize_cuda_engine(handle.read())
        if engine is None:
            raise RuntimeError(f"failed to deserialize TensorRT engine: {artifact_path}")
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError(f"failed to create TensorRT execution context: {artifact_path}")

        input_names: list[str] = []
        output_names: list[str] = []
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                input_names.append(name)
            else:
                output_names.append(name)
        if not input_names or not output_names:
            raise RuntimeError(f"TensorRT engine missing input/output tensors: {artifact_path}")
        if len(input_names) != 1:
            raise RuntimeError(
                "NovaSight TensorRT runtime requires exactly one input tensor, "
                f"got {input_names}"
            )

        input_name = input_names[0]
        engine_input_shape = _shape_tuple(engine.get_tensor_shape(input_name))
        if len(engine_input_shape) != 4:
            raise RuntimeError(f"unsupported TensorRT input shape: {engine_input_shape}")
        input_shape, input_shape_source, input_profile_shapes = _resolve_input_shape(
            engine,
            input_name=input_name,
            engine_shape=engine_input_shape,
            requested_shape=requested_shape,
        )
        input_dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(input_name)))
        if -1 in engine_input_shape:
            context.set_input_shape(input_name, input_shape)
            resolved_context_shape = _shape_tuple(context.get_tensor_shape(input_name))
            if (
                resolved_context_shape
                and len(resolved_context_shape) == 4
                and all(item > 0 for item in resolved_context_shape)
            ):
                input_shape = resolved_context_shape
        batch, channels, height, width = [int(item) for item in input_shape]
        if channels != 3:
            raise RuntimeError(f"unsupported TensorRT input channels: {input_shape}")
        self._engine_input_shape = engine_input_shape
        self._input_profile_shapes = input_profile_shapes
        self._input_shape_source = input_shape_source
        self._input_shape = TensorInputShape(
            batch=batch,
            channels=channels,
            height=height,
            width=width,
            dtype=str(input_dtype),
        )
        self._input_dtype = self._input_shape.dtype

        self._engine = engine
        self._context = context
        self._input_name = input_name
        input_allocation = _allocate_host_array(
            cudart,
            shape=(batch, channels, height, width),
            dtype=input_dtype,
        )
        self._host_allocations.append(input_allocation)
        self._host_input = input_allocation.array
        err, device_input = cudart.cudaMalloc(self._host_input.nbytes)
        _cuda_check(err, "cudaMalloc input")
        err, stream = cudart.cudaStreamCreate()
        _cuda_check(err, "cudaStreamCreate")
        self._device_input = int(device_input)
        self._stream = stream
        context.set_tensor_address(input_name, int(device_input))

        for name in output_names:
            shape = tuple(context.get_tensor_shape(name))
            if any(int(item) < 0 for item in shape):
                raise RuntimeError(f"unresolved TensorRT output shape for {name}: {shape}")
            self._output_shapes[name] = tuple(int(item) for item in shape)
            dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
            self._output_dtypes[name] = str(dtype)
            host_allocation = _allocate_host_array(
                cudart,
                shape=(int(np.prod(shape)),),
                dtype=dtype,
            )
            self._host_allocations.append(host_allocation)
            host_output = host_allocation.array
            nbytes = host_output.nbytes
            err, device_output = cudart.cudaMalloc(nbytes)
            _cuda_check(err, f"cudaMalloc output {name}")
            self._device_outputs[name] = int(device_output)
            self._host_outputs[name] = host_output
            context.set_tensor_address(name, int(device_output))

        self._output_name, self._output_candidate_columns = _select_detection_output_name(
            output_names,
            self._output_shapes,
            class_count=len(self._classes),
        )
        if self._output_name is None:
            raise RuntimeError(
                "unsupported TensorRT detection output contract; "
                f"decoder expects one NxC/CxN tensor with at least 5 columns, "
                f"outputs={self._output_shapes}"
            )
        self._output_shape = self._output_shapes[self._output_name]
        self._output_dtype = self._output_dtypes.get(self._output_name, "")
        self._io_tensors = [
            {
                "name": input_name,
                "shape": list(input_shape),
                "engine_shape": list(engine_input_shape),
                "dtype": str(input_dtype),
                "mode": "input",
            },
            *[
                {
                    "name": name,
                    "shape": list(self._output_shapes[name]),
                    "engine_shape": list(_shape_tuple(engine.get_tensor_shape(name))),
                    "dtype": self._output_dtypes[name],
                    "mode": "output",
                }
                for name in output_names
            ],
        ]
        logger.info(
            "TensorRT engine loaded path=%s engine_input=%s selected_input=%s input_source=%s profile=%s output=%s shape=%s dtype=%s outputs=%s",
            artifact_path,
            "x".join(str(item) for item in self._engine_input_shape),
            self._input_shape,
            self._input_shape_source,
            {
                name: "x".join(str(item) for item in shape)
                for name, shape in self._input_profile_shapes.items()
            },
            self._output_name,
            self._output_shape,
            self._output_dtype,
            {
                name: {
                    "shape": self._output_shapes[name],
                    "dtype": self._output_dtypes[name],
                }
                for name in output_names
            },
        )

    def _execute(self, tensor: Any) -> tuple[list[InferenceDetection], dict[str, Any]]:
        if isinstance(tensor, DeviceTensor):
            return self._execute_device_tensor(tensor)
        return self._execute_host_tensor(tensor)

    def _execute_host_tensor(self, tensor: Any) -> tuple[list[InferenceDetection], dict[str, Any]]:
        import numpy as np

        if (
            self._cudart is None
            or self._context is None
            or self._host_input is None
            or not self._host_outputs
            or self._device_input is None
            or self._stream is None
            or not self._output_name
        ):
            raise RuntimeError("TensorRT execution buffers are not initialized")
        cudart = self._cudart
        host_output = self._host_outputs.get(self._output_name)
        if host_output is None:
            raise RuntimeError(f"TensorRT selected output buffer is not initialized: {self._output_name}")
        timings: dict[str, float] = {}
        prepare_host_start_ns = time.monotonic_ns()
        host_input = np.ascontiguousarray(tensor, dtype=self._host_input.dtype)
        if host_input.shape != self._host_input.shape:
            raise RuntimeError(
                f"TensorRT input tensor shape mismatch: got {host_input.shape}, "
                f"expected {self._host_input.shape}"
            )
        np.copyto(self._host_input, host_input)
        h2d_start_ns = time.monotonic_ns()
        h2d = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        d2h = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        _cuda_check(
            cudart.cudaMemcpyAsync(
                self._device_input,
                self._host_input.ctypes.data,
                self._host_input.nbytes,
                h2d,
                self._stream,
            ),
            "H2D",
        )
        execute_start_ns = time.monotonic_ns()
        ok = self._context.execute_async_v3(int(self._stream))
        if ok is False:
            raise RuntimeError("TensorRT execute_async_v3 returned false")
        d2h_start_ns = time.monotonic_ns()
        _cuda_check(
            cudart.cudaMemcpyAsync(
                host_output.ctypes.data,
                self._device_outputs[self._output_name],
                host_output.nbytes,
                d2h,
                self._stream,
            ),
            "D2H",
        )
        sync_start_ns = time.monotonic_ns()
        _cuda_check(cudart.cudaStreamSynchronize(self._stream), "stream synchronize")
        decode_start_ns = time.monotonic_ns()
        output = host_output.reshape(self._output_shape).astype(np.float32, copy=False)
        decode_debug: dict[str, Any] = {}
        detections = decode_nx6_detections(
            output,
            confidence_threshold=self.confidence_threshold,
            nms_threshold=self.nms_threshold,
            class_count=len(self._classes),
            debug=decode_debug,
        )
        done_ns = time.monotonic_ns()
        timings.update(
            {
                "input_location": "host",
                "host_prepare_ms": _elapsed_ms(prepare_host_start_ns, h2d_start_ns),
                "h2d_enqueue_ms": _elapsed_ms(h2d_start_ns, execute_start_ns),
                "execute_enqueue_ms": _elapsed_ms(execute_start_ns, d2h_start_ns),
                "d2h_enqueue_ms": _elapsed_ms(d2h_start_ns, sync_start_ns),
                "stream_sync_ms": _elapsed_ms(sync_start_ns, decode_start_ns),
                "decode_ms": _elapsed_ms(decode_start_ns, done_ns),
                "total_ms": _elapsed_ms(prepare_host_start_ns, done_ns),
            }
        )
        decode_debug["timings"] = timings
        return detections, decode_debug

    def _execute_device_tensor(self, tensor: DeviceTensor) -> tuple[list[InferenceDetection], dict[str, Any]]:
        if (
            self._cudart is None
            or self._context is None
            or self._host_input is None
            or not self._host_outputs
            or self._device_input is None
            or self._stream is None
            or not self._input_name
            or not self._output_name
        ):
            raise RuntimeError("TensorRT execution buffers are not initialized")
        expected_shape = tuple(int(item) for item in self._host_input.shape)
        if tuple(int(item) for item in tensor.shape) != expected_shape:
            raise RuntimeError(
                f"TensorRT device tensor shape mismatch: got {tensor.shape}, "
                f"expected {expected_shape}"
            )
        expected_dtype = normalize_tensor_dtype(getattr(self._host_input, "dtype", self._input_dtype))
        try:
            actual_dtype = normalize_tensor_dtype(tensor.dtype)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        if actual_dtype != expected_dtype:
            raise RuntimeError(
                f"TensorRT device tensor dtype mismatch: got {tensor.dtype}, expected {expected_dtype}"
            )
        if int(tensor.nbytes) < int(self._host_input.nbytes):
            raise RuntimeError(
                f"TensorRT device tensor is too small: got {tensor.nbytes} bytes, "
                f"expected at least {self._host_input.nbytes}"
            )
        bind_start_ns = time.monotonic_ns()
        self._context.set_tensor_address(self._input_name, int(tensor.device_ptr))
        bind_done_ns = time.monotonic_ns()
        owner_release_status = "not_present"
        owner_release_token = _device_tensor_owner_release_token(tensor)
        try:
            detections, decode_debug = self._execute_bound_input(bind_start_ns)
        finally:
            try:
                self._context.set_tensor_address(self._input_name, int(self._device_input))
            finally:
                owner_release_status = _release_device_tensor_owner(tensor)
        timings = decode_debug.setdefault("timings", {})
        timings["input_location"] = "device"
        timings["device_bind_ms"] = _elapsed_ms(bind_start_ns, bind_done_ns)
        timings["h2d_enqueue_ms"] = 0.0
        timings["host_prepare_ms"] = 0.0
        timings["device_owner_release"] = owner_release_status
        timings["device_owner_release_token"] = owner_release_token
        return detections, decode_debug

    def _execute_bound_input(self, start_ns: int) -> tuple[list[InferenceDetection], dict[str, Any]]:
        import numpy as np

        if (
            self._cudart is None
            or self._context is None
            or not self._host_outputs
            or self._stream is None
            or not self._output_name
        ):
            raise RuntimeError("TensorRT execution buffers are not initialized")
        cudart = self._cudart
        host_output = self._host_outputs.get(self._output_name)
        if host_output is None:
            raise RuntimeError(f"TensorRT selected output buffer is not initialized: {self._output_name}")
        if self._output_name not in self._device_outputs:
            raise RuntimeError(f"TensorRT selected device output is not initialized: {self._output_name}")
        d2h = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        execute_start_ns = time.monotonic_ns()
        ok = self._context.execute_async_v3(int(self._stream))
        if ok is False:
            raise RuntimeError("TensorRT execute_async_v3 returned false")
        d2h_start_ns = time.monotonic_ns()
        _cuda_check(
            cudart.cudaMemcpyAsync(
                host_output.ctypes.data,
                self._device_outputs[self._output_name],
                host_output.nbytes,
                d2h,
                self._stream,
            ),
            "D2H",
        )
        sync_start_ns = time.monotonic_ns()
        _cuda_check(cudart.cudaStreamSynchronize(self._stream), "stream synchronize")
        decode_start_ns = time.monotonic_ns()
        output = host_output.reshape(self._output_shape).astype(np.float32, copy=False)
        decode_debug: dict[str, Any] = {}
        detections = decode_nx6_detections(
            output,
            confidence_threshold=self.confidence_threshold,
            nms_threshold=self.nms_threshold,
            class_count=len(self._classes),
            debug=decode_debug,
        )
        done_ns = time.monotonic_ns()
        timings = {
            "execute_enqueue_ms": _elapsed_ms(execute_start_ns, d2h_start_ns),
            "d2h_enqueue_ms": _elapsed_ms(d2h_start_ns, sync_start_ns),
            "stream_sync_ms": _elapsed_ms(sync_start_ns, decode_start_ns),
            "decode_ms": _elapsed_ms(decode_start_ns, done_ns),
            "total_ms": _elapsed_ms(start_ns, done_ns),
        }
        decode_debug["timings"] = timings
        return detections, decode_debug

    def close(self) -> None:
        cudart = self._cudart
        if cudart is not None:
            if self._stream is not None:
                try:
                    cudart.cudaStreamSynchronize(self._stream)
                except Exception:
                    pass
            if self._device_input is not None:
                try:
                    cudart.cudaFree(self._device_input)
                except Exception:
                    pass
            for pointer in self._device_outputs.values():
                try:
                    cudart.cudaFree(pointer)
                except Exception:
                    pass
            if self._stream is not None:
                try:
                    cudart.cudaStreamDestroy(self._stream)
                except Exception:
                    pass
        for allocation in self._host_allocations:
            allocation.release()
        self._device_input = None
        self._device_outputs = {}
        self._stream = None
        self._context = None
        self._engine = None
        self._host_input = None
        self._host_allocations = []
        self._host_outputs = {}
        self._output_shapes = {}
        self._output_dtypes = {}
        self._output_shape = ()
        self._output_dtype = ""
        self._output_candidate_columns = None
        self._input_dtype = "float32"
        self._input_name = ""
        self._output_name = ""
        self._engine_input_shape = ()
        self._input_profile_shapes = {}
        self._input_shape_source = ""
        self._io_tensors = []
        self._loaded = False
        self._warmed = False

    def _log_slow_inference(self, timings: dict[str, float], decode_debug: dict[str, Any]) -> None:
        total_ms = float(timings.get("total_ms") or 0.0)
        if total_ms < 100.0:
            return
        now_ns = time.monotonic_ns()
        if now_ns - self._last_slow_log_ns < 1_000_000_000:
            return
        self._last_slow_log_ns = now_ns
        logger.warning(
            "TensorRT slow inference input=%s engine_input=%s source=%s output=%s timings=%s decode_timings=%s",
            self._input_shape,
            "x".join(str(item) for item in self._engine_input_shape),
            self._input_shape_source,
            self._output_shape,
            timings,
            decode_debug.get("timings", {}),
        )

    def _log_failure_once(self, reason: str, *, with_trace: bool = False) -> None:
        if reason == self._last_failure_logged:
            return
        self._last_failure_logged = reason
        if with_trace:
            logger.exception("TensorRT inference failed: %s", reason)
        else:
            logger.warning("TensorRT inference failed: %s", reason)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _host_input_pinned(self) -> bool:
        return bool(self._host_allocations and self._host_allocations[0].pinned)

    def _host_output_pinned(self) -> bool:
        return any(allocation.pinned for allocation in self._host_allocations[1:])


TensorRtGpuInferBackend = TensorRtInferenceEngine


@dataclass
class _HostAllocation:
    array: Any
    pointer: int = 0
    pinned: bool = False
    _release: Any | None = None
    _owner: Any | None = None
    _released: bool = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        if not self.pinned or self._release is None or self.pointer <= 0:
            return
        try:
            self._release(self.pointer)
        except Exception:
            pass


def _allocate_host_array(
    cudart: Any,
    *,
    shape: tuple[int, ...],
    dtype: Any,
) -> _HostAllocation:
    import numpy as np

    array_shape = tuple(int(item) for item in shape)
    np_dtype = np.dtype(dtype)
    nbytes = int(np.prod(array_shape)) * int(np_dtype.itemsize)
    pinned = _try_allocate_pinned_host_array(
        cudart,
        shape=array_shape,
        dtype=np_dtype,
        nbytes=nbytes,
        np=np,
    )
    if pinned is not None:
        return pinned
    array = np.empty(array_shape, dtype=np_dtype)
    return _HostAllocation(
        array=array,
        pointer=int(array.ctypes.data),
        pinned=False,
    )


def _try_allocate_pinned_host_array(
    cudart: Any,
    *,
    shape: tuple[int, ...],
    dtype: Any,
    nbytes: int,
    np: Any,
) -> _HostAllocation | None:
    releaser = getattr(cudart, "cudaFreeHost", None)
    if not callable(releaser):
        return None
    for allocator_name in ("cudaHostAlloc", "cudaMallocHost"):
        allocator = getattr(cudart, allocator_name, None)
        if not callable(allocator):
            continue
        try:
            if allocator_name == "cudaHostAlloc":
                flags = int(getattr(cudart, "cudaHostAllocDefault", 0) or 0)
                result = allocator(int(nbytes), flags)
            else:
                result = allocator(int(nbytes))
            err, pointer = _cuda_result_error_pointer(result)
        except Exception:
            continue
        if err != 0 or pointer <= 0:
            continue
        ctypes_array = (ctypes.c_uint8 * int(nbytes)).from_address(pointer)
        array = np.ctypeslib.as_array(ctypes_array).view(dtype).reshape(shape)
        return _HostAllocation(
            array=array,
            pointer=pointer,
            pinned=True,
            _release=releaser,
            _owner=ctypes_array,
        )
    return None


def _cuda_result_error_pointer(result: Any) -> tuple[int, int]:
    if isinstance(result, tuple):
        if len(result) < 2:
            return int(result[0]), 0
        return int(result[0]), int(result[1])
    return int(result), 0


def _cuda_check(result: Any, what: str) -> None:
    err = result[0] if isinstance(result, tuple) else result
    if int(err) != 0:
        raise RuntimeError(f"CUDA error {int(err)} at {what}")


def _resolve_input_shape(
    engine: Any,
    *,
    input_name: str,
    engine_shape: tuple[int, ...],
    requested_shape: tuple[int, int, int, int] | None = None,
) -> tuple[tuple[int, ...], str, dict[str, tuple[int, ...]]]:
    del requested_shape
    if -1 not in engine_shape:
        return engine_shape, "engine_static", {}

    profile_shapes = _read_input_profile_shapes(engine, input_name)
    if profile_shapes:
        profile_opt = profile_shapes.get("opt", ())
        if _shape_is_static(profile_opt):
            return profile_opt, "engine_profile_opt", profile_shapes
    raise RuntimeError(
        f"TensorRT dynamic input shape requires a static optimization profile opt shape, "
        f"got engine_shape={engine_shape} input={input_name} profile={profile_shapes}"
    )


def _read_input_profile_shapes(engine: Any, input_name: str) -> dict[str, tuple[int, ...]]:
    shapes: Any | None = None
    try:
        getter = getattr(engine, "get_tensor_profile_shape", None)
        if callable(getter):
            shapes = getter(input_name, 0)
    except Exception:
        shapes = None
    if shapes is None:
        try:
            getter = getattr(engine, "get_profile_shape", None)
            if callable(getter):
                shapes = getter(0, input_name)
        except Exception:
            shapes = None
    if shapes is None or len(shapes) != 3:
        return {}
    profile_min, profile_opt, profile_max = shapes
    return {
        "min": _shape_tuple(profile_min),
        "opt": _shape_tuple(profile_opt),
        "max": _shape_tuple(profile_max),
    }


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    try:
        return tuple(int(item) for item in value)
    except TypeError:
        dims = getattr(value, "d", None)
        nb_dims = int(getattr(value, "nbDims", 0) or 0)
        if dims is None or nb_dims <= 0:
            return ()
        return tuple(int(dims[index]) for index in range(nb_dims))


def _shape_is_static(shape: tuple[int, ...]) -> bool:
    return bool(shape) and all(int(item) > 0 for item in shape)


def _elapsed_ms(start_ns: int, end_ns: int) -> float:
    return max(0.0, (end_ns - start_ns) / 1e6)


def _release_device_tensor_owner(tensor: DeviceTensor) -> str:
    owner = getattr(tensor, "owner", None)
    release = getattr(owner, "release", None)
    if not callable(release):
        return "not_present"
    try:
        release()
    except Exception as exc:
        logger.warning("TensorRT device tensor owner release failed: %s", exc)
        return "failed"
    return "released"


def _device_tensor_owner_release_token(tensor: DeviceTensor) -> int | None:
    owner = getattr(tensor, "owner", None)
    if owner is None:
        return None
    for attr in ("release_token", "token", "_token"):
        value = getattr(owner, attr, None)
        if callable(value):
            try:
                value = value()
            except Exception:
                continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None
