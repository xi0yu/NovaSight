from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .contracts import (
    EngineInspectionErrorCode,
    EngineInspectionResult,
    OptimizationProfileDescriptor,
    ShapeRange,
    TensorDescriptor,
)


@dataclass
class _LoadedEngine:
    engine: Any
    trt: Any
    runtime: Any | None = None

    def close(self) -> None:
        self.engine = None
        self.runtime = None


Deserializer = Callable[[Path], tuple[Any, Any] | _LoadedEngine]


class EngineInspector:
    """Read the immutable TensorRT engine contract without starting inference."""

    def __init__(self, *, deserialize: Deserializer | None = None) -> None:
        self._deserialize = deserialize or self._deserialize_with_tensorrt

    def inspect(self, engine_path: Path) -> EngineInspectionResult:
        path = Path(engine_path)
        file_error = self._validate_file(path)
        if file_error is not None:
            return file_error
        try:
            initial_signature = (path.stat().st_size, path.stat().st_mtime_ns)
        except OSError as exc:
            return self._failure(
                EngineInspectionErrorCode.FILE_NOT_READABLE,
                f"TensorRT engine file cannot be stated: {exc}",
            )

        loaded: _LoadedEngine | None = None
        try:
            raw_loaded = self._deserialize(path)
            loaded = (
                raw_loaded
                if isinstance(raw_loaded, _LoadedEngine)
                else _LoadedEngine(engine=raw_loaded[0], trt=raw_loaded[1])
            )
            if loaded.engine is None:
                return self._failure(
                    EngineInspectionErrorCode.DESERIALIZE_FAILED,
                    "deserializeCudaEngine returned null",
                )
            result = self._inspect_loaded_engine(loaded.engine, loaded.trt)
            try:
                final_signature = (path.stat().st_size, path.stat().st_mtime_ns)
            except OSError as exc:
                return self._failure(
                    EngineInspectionErrorCode.FILE_NOT_READABLE,
                    f"TensorRT engine file changed during inspection: {exc}",
                )
            if final_signature != initial_signature:
                return self._failure(
                    EngineInspectionErrorCode.FILE_NOT_READABLE,
                    "TensorRT engine file changed while it was being inspected",
                )
            return result
        except Exception as exc:
            return self._failure(self._classify_exception(exc), str(exc))
        finally:
            if loaded is not None:
                loaded.close()

    @staticmethod
    def _validate_file(path: Path) -> EngineInspectionResult | None:
        if not path.exists():
            return EngineInspector._failure(
                EngineInspectionErrorCode.FILE_NOT_FOUND,
                f"TensorRT engine file does not exist: {path}",
            )
        if not path.is_file():
            return EngineInspector._failure(
                EngineInspectionErrorCode.FILE_NOT_READABLE,
                f"TensorRT engine path is not a file: {path}",
            )
        try:
            size = path.stat().st_size
            with path.open("rb") as handle:
                handle.read(1)
        except OSError as exc:
            return EngineInspector._failure(
                EngineInspectionErrorCode.FILE_NOT_READABLE,
                f"TensorRT engine file is not readable: {exc}",
            )
        if size <= 0:
            return EngineInspector._failure(
                EngineInspectionErrorCode.EMPTY_ENGINE,
                f"TensorRT engine file is empty: {path}",
            )
        return None

    def _inspect_loaded_engine(self, engine: Any, trt: Any) -> EngineInspectionResult:
        tensors: list[TensorDescriptor] = []
        input_mode = getattr(getattr(trt, "TensorIOMode", None), "INPUT", None)
        io_count = int(getattr(engine, "num_io_tensors", 0) or 0)
        for index in range(io_count):
            name = str(engine.get_tensor_name(index))
            raw_mode = engine.get_tensor_mode(name)
            mode = "input" if raw_mode == input_mode or _enum_text(raw_mode) == "input" else "output"
            tensors.append(
                TensorDescriptor(
                    name=name,
                    io_mode=mode,
                    engine_shape=_shape_tuple(engine.get_tensor_shape(name)),
                    data_type=_normalize_dtype(engine.get_tensor_dtype(name)),
                    tensor_format=_enum_text(engine.get_tensor_format(name)),
                    is_shape_tensor=bool(engine.is_shape_inference_io(name)),
                    bytes_per_component=int(engine.get_tensor_bytes_per_component(name)),
                    components_per_element=int(engine.get_tensor_components_per_element(name)),
                    vectorized_dim=int(engine.get_tensor_vectorized_dim(name)),
                )
            )

        inputs = tuple(tensor for tensor in tensors if tensor.io_mode == "input")
        outputs = tuple(tensor for tensor in tensors if tensor.io_mode == "output")
        profile_count = int(getattr(engine, "num_optimization_profiles", 0) or 0)
        profiles = tuple(
            self._read_profile(engine, inputs, profile_index)
            for profile_index in range(profile_count)
        )
        has_dynamic_shape = any(
            any(dimension < 0 for dimension in tensor.engine_shape)
            for tensor in tensors
        )
        has_shape_input = any(tensor.is_shape_tensor for tensor in inputs)
        compatibility_error = self._compatibility_error(
            inputs=inputs,
            outputs=outputs,
            profiles=profiles,
        )
        return EngineInspectionResult(
            deserialize_ok=True,
            compatible=compatibility_error is None,
            engine_name=str(getattr(engine, "name", "") or ""),
            inputs=inputs,
            outputs=outputs,
            profiles=profiles,
            has_dynamic_shape=has_dynamic_shape,
            has_shape_input=has_shape_input,
            error_code=compatibility_error[0] if compatibility_error else None,
            raw_error=compatibility_error[1] if compatibility_error else "",
        )

    @staticmethod
    def _read_profile(
        engine: Any,
        inputs: tuple[TensorDescriptor, ...],
        profile_index: int,
    ) -> OptimizationProfileDescriptor:
        ranges: dict[str, ShapeRange] = {}
        for tensor in inputs:
            shapes: Any | None = None
            getter = getattr(engine, "get_tensor_profile_shape", None)
            if callable(getter):
                try:
                    shapes = getter(tensor.name, profile_index)
                except TypeError:
                    shapes = getter(profile_index, tensor.name)
            if shapes is None:
                getter = getattr(engine, "get_profile_shape", None)
                if callable(getter):
                    shapes = getter(profile_index, tensor.name)
            if shapes is None or len(shapes) != 3:
                continue
            minimum, optimum, maximum = shapes
            ranges[tensor.name] = ShapeRange(
                minimum=_shape_tuple(minimum),
                optimum=_shape_tuple(optimum),
                maximum=_shape_tuple(maximum),
            )
        return OptimizationProfileDescriptor(
            profile_index=profile_index,
            input_ranges=ranges,
        )

    @staticmethod
    def _compatibility_error(
        *,
        inputs: tuple[TensorDescriptor, ...],
        outputs: tuple[TensorDescriptor, ...],
        profiles: tuple[OptimizationProfileDescriptor, ...],
    ) -> tuple[EngineInspectionErrorCode, str] | None:
        if len(inputs) != 1 or not outputs:
            return (
                EngineInspectionErrorCode.UNSUPPORTED_IO_COUNT,
                "NovaSight requires exactly one image input and at least one output "
                f"(inputs={len(inputs)}, outputs={len(outputs)})",
            )
        input_tensor = inputs[0]
        if input_tensor.is_shape_tensor:
            return (
                EngineInspectionErrorCode.UNSUPPORTED_DYNAMIC_SHAPE,
                f"NovaSight does not support shape tensor input: {input_tensor.name}",
            )
        if len(input_tensor.engine_shape) != 4:
            return (
                EngineInspectionErrorCode.UNSUPPORTED_INPUT_RANK,
                f"NovaSight requires a four-dimensional image input, got {input_tensor.engine_shape}",
            )
        if input_tensor.tensor_format not in {"linear", "chw"}:
            return (
                EngineInspectionErrorCode.UNKNOWN_TENSOR_FORMAT,
                f"NovaSight supports linear NCHW input, got {input_tensor.tensor_format or '<unknown>'}",
            )
        runtime_shape = input_tensor.engine_shape
        if any(dimension < 0 for dimension in runtime_shape):
            first_range = profiles[0].input_ranges.get(input_tensor.name) if profiles else None
            if first_range is None or not _is_concrete_shape(first_range.optimum):
                return (
                    EngineInspectionErrorCode.UNSUPPORTED_DYNAMIC_SHAPE,
                    "dynamic TensorRT input requires a concrete optimization profile OPT shape",
                )
            runtime_shape = first_range.optimum
        if runtime_shape[0] != 1:
            return (
                EngineInspectionErrorCode.UNSUPPORTED_DYNAMIC_SHAPE,
                f"NovaSight production inference requires batch=1, got {runtime_shape}",
            )
        if runtime_shape[1] != 3:
            return (
                EngineInspectionErrorCode.UNSUPPORTED_LAYOUT,
                f"NovaSight requires NCHW three-channel input, got {runtime_shape}",
            )
        return None

    @staticmethod
    def _deserialize_with_tensorrt(path: Path) -> _LoadedEngine:
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with path.open("rb") as handle:
            engine = runtime.deserialize_cuda_engine(handle.read())
        return _LoadedEngine(engine=engine, trt=trt, runtime=runtime)

    @staticmethod
    def _classify_exception(exc: Exception) -> EngineInspectionErrorCode:
        message = str(exc).lower()
        if "plugin" in message:
            return EngineInspectionErrorCode.PLUGIN_MISSING
        if "out of memory" in message or "cuda_error_out_of_memory" in message:
            return EngineInspectionErrorCode.OUT_OF_MEMORY
        if "deserialize" in message or "serialization" in message:
            return EngineInspectionErrorCode.DESERIALIZE_FAILED
        return EngineInspectionErrorCode.INTERNAL_ERROR

    @staticmethod
    def _failure(
        code: EngineInspectionErrorCode,
        raw_error: str,
    ) -> EngineInspectionResult:
        return EngineInspectionResult(
            deserialize_ok=False,
            compatible=False,
            requires_plugin=code is EngineInspectionErrorCode.PLUGIN_MISSING,
            error_code=code,
            raw_error=raw_error,
        )


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in value)
    except TypeError:
        dims = getattr(value, "d", None)
        count = int(getattr(value, "nbDims", 0) or 0)
        if dims is None or count <= 0:
            return ()
        return tuple(int(dims[index]) for index in range(count))


def _enum_text(value: Any) -> str:
    text = str(getattr(value, "name", value) or "").strip().lower()
    return text.rsplit(".", 1)[-1]


def _normalize_dtype(value: Any) -> str:
    text = _enum_text(value).replace("_", "")
    aliases = {
        "float": "float32",
        "fp32": "float32",
        "float32": "float32",
        "half": "float16",
        "fp16": "float16",
        "float16": "float16",
        "int8": "int8",
        "int32": "int32",
        "int64": "int64",
        "bool": "bool",
        "uint8": "uint8",
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
    }
    return aliases.get(text, text)


def _is_concrete_shape(shape: tuple[int, ...]) -> bool:
    return bool(shape) and all(int(dimension) > 0 for dimension in shape)


__all__ = ["EngineInspector"]
