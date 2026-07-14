from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class EngineInspectionErrorCode(str, Enum):
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    FILE_NOT_READABLE = "FILE_NOT_READABLE"
    EMPTY_ENGINE = "EMPTY_ENGINE"
    DESERIALIZE_FAILED = "DESERIALIZE_FAILED"
    PLUGIN_MISSING = "PLUGIN_MISSING"
    UNSUPPORTED_IO_COUNT = "UNSUPPORTED_IO_COUNT"
    UNSUPPORTED_INPUT_RANK = "UNSUPPORTED_INPUT_RANK"
    UNSUPPORTED_LAYOUT = "UNSUPPORTED_LAYOUT"
    UNSUPPORTED_DYNAMIC_SHAPE = "UNSUPPORTED_DYNAMIC_SHAPE"
    UNKNOWN_TENSOR_FORMAT = "UNKNOWN_TENSOR_FORMAT"
    OUT_OF_MEMORY = "OUT_OF_MEMORY"
    INTERNAL_ERROR = "INTERNAL_ERROR"


@dataclass(frozen=True)
class TensorDescriptor:
    name: str
    io_mode: str
    engine_shape: tuple[int, ...]
    data_type: str
    tensor_format: str
    is_shape_tensor: bool
    bytes_per_component: int
    components_per_element: int
    vectorized_dim: int


@dataclass(frozen=True)
class ShapeRange:
    minimum: tuple[int, ...]
    optimum: tuple[int, ...]
    maximum: tuple[int, ...]


@dataclass(frozen=True)
class OptimizationProfileDescriptor:
    profile_index: int
    input_ranges: dict[str, ShapeRange] = field(default_factory=dict)


@dataclass(frozen=True)
class EngineInspectionResult:
    deserialize_ok: bool
    compatible: bool
    engine_name: str = ""
    inputs: tuple[TensorDescriptor, ...] = ()
    outputs: tuple[TensorDescriptor, ...] = ()
    profiles: tuple[OptimizationProfileDescriptor, ...] = ()
    has_dynamic_shape: bool = False
    has_shape_input: bool = False
    requires_plugin: bool = False
    error_code: EngineInspectionErrorCode | None = None
    raw_error: str = ""
    warnings: tuple[str, ...] = ()


__all__ = [
    "EngineInspectionErrorCode",
    "EngineInspectionResult",
    "OptimizationProfileDescriptor",
    "ShapeRange",
    "TensorDescriptor",
]
