from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from .contracts import InferenceDetection, InferenceResult
from .input import PreparedTensorInput, TensorInputShape, parse_tensor_input_shape, prepare_tensor_input
from .onnxruntime_engine import (
    _prepare_numpy_tensor,
    _preprocess_debug,
    _scale_detections_to_input_frame,
    decode_nx6_detections,
)


logger = logging.getLogger("novasight.inference.tensorrt")


class TensorRtInferenceEngine:
    engine_id = "tensorrt"

    def __init__(
        self,
        *,
        confidence_threshold: float = 0.25,
        nms_threshold: float = 0.45,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
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
        self._device_outputs: dict[str, int] = {}
        self._host_outputs: dict[str, Any] = {}
        self._output_shapes: dict[str, tuple[int, ...]] = {}
        self._output_dtypes: dict[str, str] = {}
        self._output_shape: tuple[int, ...] = ()
        self._output_dtype = ""
        self._last_failure_logged = ""
        self._engine_input_shape: tuple[int, ...] = ()
        self._input_profile_shapes: dict[str, tuple[int, ...]] = {}
        self._input_shape_source = ""
        self._last_slow_log_ns = 0

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
            "reason": self._reason,
            "input_shape": str(self._input_shape) if self._input_shape is not None else "",
            "output_shape": "x".join(str(item) for item in self._output_shape),
            "output_name": self._output_name,
            "output_dtype": self._output_dtype,
            "engine_input_shape": "x".join(str(item) for item in self._engine_input_shape),
            "input_shape_source": self._input_shape_source,
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
            "confidence_threshold": self.confidence_threshold,
            "nms_threshold": self.nms_threshold,
            "last_input_mode": self._last_input.mode if self._last_input is not None else "",
            "last_input_needs_resize": (
                self._last_input.needs_resize if self._last_input is not None else False
            ),
        }

    def load(self, artifact_path: Path, classes: list[str], input_shape: str) -> None:
        if not artifact_path.suffix == ".engine":
            raise ValueError(f"TensorRT artifact must be .engine: {artifact_path}")
        if not classes:
            raise ValueError("TensorRT classes must not be empty")
        if not input_shape.strip():
            raise ValueError("TensorRT input shape must not be empty")
        parsed_shape = parse_tensor_input_shape(input_shape)
        if not self.available():
            raise RuntimeError(self._reason)
        self.close()
        self._classes = list(classes)
        self._last_input = None
        self._load_engine(artifact_path, parsed_shape)
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
            tensor = _prepare_numpy_tensor(self._last_input, self._input_shape)
            execute_start_ns = time.monotonic_ns()
            detections, decode_debug = self._execute(tensor)
            scale_start_ns = time.monotonic_ns()
            detections = _scale_detections_to_input_frame(
                detections,
                prepared=self._last_input,
                shape=self._input_shape,
            )
            preprocess_debug = _preprocess_debug(self._last_input, self._input_shape)
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
            self._log_slow_inference(timings, decode_debug)
        except ValueError as exc:
            self._log_failure_once(f"input rejected: {exc}")
            return InferenceResult(available=False, reason=str(exc))
        except Exception as exc:
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
                "preprocess": preprocess_debug,
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

    def _load_engine(self, artifact_path: Path, configured_shape: TensorInputShape) -> None:
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

        input_name = input_names[0]
        engine_input_shape = _shape_tuple(engine.get_tensor_shape(input_name))
        if len(engine_input_shape) != 4:
            raise RuntimeError(f"unsupported TensorRT input shape: {engine_input_shape}")
        configured_tuple = (
            configured_shape.batch,
            configured_shape.channels,
            configured_shape.height,
            configured_shape.width,
        )
        input_shape, input_shape_source, input_profile_shapes = _resolve_input_shape(
            engine,
            input_name=input_name,
            engine_shape=engine_input_shape,
            configured_shape=configured_tuple,
        )
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
        )

        self._engine = engine
        self._context = context
        self._input_name = input_name
        self._host_input = np.empty((batch, channels, height, width), dtype=np.float32)
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
            host_output = np.empty(int(np.prod(shape)), dtype=dtype)
            nbytes = host_output.nbytes
            err, device_output = cudart.cudaMalloc(nbytes)
            _cuda_check(err, f"cudaMalloc output {name}")
            self._device_outputs[name] = int(device_output)
            self._host_outputs[name] = host_output
            context.set_tensor_address(name, int(device_output))

        self._output_name = next(
            (name for name in output_names if len(self._output_shapes[name]) == 3),
            output_names[0],
        )
        self._output_shape = self._output_shapes[self._output_name]
        self._output_dtype = self._output_dtypes.get(self._output_name, "")
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
        host_input = np.ascontiguousarray(tensor, dtype=np.float32)
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
        self._device_input = None
        self._device_outputs = {}
        self._stream = None
        self._context = None
        self._engine = None
        self._host_input = None
        self._host_outputs = {}
        self._output_shapes = {}
        self._output_dtypes = {}
        self._output_shape = ()
        self._output_dtype = ""
        self._output_name = ""
        self._engine_input_shape = ()
        self._input_profile_shapes = {}
        self._input_shape_source = ""
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


def _cuda_check(result: Any, what: str) -> None:
    err = result[0] if isinstance(result, tuple) else result
    if int(err) != 0:
        raise RuntimeError(f"CUDA error {int(err)} at {what}")


def _resolve_input_shape(
    engine: Any,
    *,
    input_name: str,
    engine_shape: tuple[int, ...],
    configured_shape: tuple[int, ...],
) -> tuple[tuple[int, ...], str, dict[str, tuple[int, ...]]]:
    if -1 not in engine_shape:
        return engine_shape, "engine_static", {}

    profile_shapes = _read_input_profile_shapes(engine, input_name)
    if profile_shapes:
        profile_min = profile_shapes.get("min", ())
        profile_opt = profile_shapes.get("opt", ())
        profile_max = profile_shapes.get("max", ())
        if _shape_is_static(profile_opt):
            return profile_opt, "engine_profile_opt", profile_shapes
        if _shape_fits_profile(configured_shape, profile_min, profile_max):
            return configured_shape, "configured_within_profile", profile_shapes

    return configured_shape, "configured_fallback", profile_shapes


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


def _shape_fits_profile(
    shape: tuple[int, ...],
    profile_min: tuple[int, ...],
    profile_max: tuple[int, ...],
) -> bool:
    if not shape or len(shape) != len(profile_min) or len(shape) != len(profile_max):
        return False
    return all(low <= item <= high for item, low, high in zip(shape, profile_min, profile_max))


def _elapsed_ms(start_ns: int, end_ns: int) -> float:
    return max(0.0, (end_ns - start_ns) / 1e6)
