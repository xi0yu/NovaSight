from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .contracts import InferenceDetection, InferenceResult
from .input import PreparedTensorInput, TensorInputShape, parse_tensor_input_shape, prepare_tensor_input
from .onnxruntime_engine import (
    _prepare_numpy_tensor,
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
        try:
            self._last_input = prepare_tensor_input(frame, self._input_shape)
            tensor = _prepare_numpy_tensor(self._last_input, self._input_shape)
            detections, decode_debug = self._execute(tensor)
            detections = _scale_detections_to_input_frame(
                detections,
                prepared=self._last_input,
                shape=self._input_shape,
            )
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
                "decode": decode_debug,
            },
        )

    def _warmup(self) -> None:
        if self._input_shape is None:
            self._warmed = False
            return
        try:
            import numpy as np

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
        input_shape = tuple(engine.get_tensor_shape(input_name))
        if len(input_shape) != 4:
            raise RuntimeError(f"unsupported TensorRT input shape: {input_shape}")
        if -1 in input_shape:
            input_shape = (
                configured_shape.batch,
                configured_shape.channels,
                configured_shape.height,
                configured_shape.width,
            )
            context.set_input_shape(input_name, input_shape)
        batch, channels, height, width = [int(item) for item in input_shape]
        if channels != 3:
            raise RuntimeError(f"unsupported TensorRT input channels: {input_shape}")
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
            "TensorRT engine loaded path=%s input=%s output=%s shape=%s dtype=%s outputs=%s",
            artifact_path,
            self._input_shape,
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
        host_input = np.ascontiguousarray(tensor, dtype=np.float32)
        if host_input.shape != self._host_input.shape:
            raise RuntimeError(
                f"TensorRT input tensor shape mismatch: got {host_input.shape}, "
                f"expected {self._host_input.shape}"
            )
        np.copyto(self._host_input, host_input)
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
        ok = self._context.execute_async_v3(int(self._stream))
        if ok is False:
            raise RuntimeError("TensorRT execute_async_v3 returned false")
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
        _cuda_check(cudart.cudaStreamSynchronize(self._stream), "stream synchronize")
        output = host_output.reshape(self._output_shape).astype(np.float32, copy=False)
        decode_debug: dict[str, Any] = {}
        detections = decode_nx6_detections(
            output,
            confidence_threshold=self.confidence_threshold,
            nms_threshold=self.nms_threshold,
            class_count=len(self._classes),
            debug=decode_debug,
        )
        return detections, decode_debug

    def close(self) -> None:
        cudart = self._cudart
        if cudart is not None:
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
        self._loaded = False
        self._warmed = False

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
