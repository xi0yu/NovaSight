from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
import hashlib
import json
from math import isfinite
from pathlib import Path

from novasight.model_registry.fingerprint import sha256_file

from .contracts import EngineInspectionResult, TensorDescriptor
from .parser_registry import get_parser_definition


MODEL_PROFILE_SCHEMA_VERSION = 1


class ModelStatus(str, Enum):
    UNINSPECTED = "UNINSPECTED"
    INSPECTING = "INSPECTING"
    NEEDS_CONFIGURATION = "NEEDS_CONFIGURATION"
    READY_FOR_PROBE = "READY_FOR_PROBE"
    PROBING = "PROBING"
    VALIDATED = "VALIDATED"
    INVALID = "INVALID"
    INCOMPATIBLE = "INCOMPATIBLE"
    ACTIVE = "ACTIVE"


@dataclass(frozen=True)
class EngineIdentity:
    path: str
    sha256: str
    file_size: int
    modified_at_ns: int


@dataclass(frozen=True)
class ProfileInput:
    name: str
    runtime_shape: tuple[int, ...]
    engine_shape: tuple[int, ...]
    dtype: str
    layout: str = "NCHW"
    profile_index: int = 0


@dataclass(frozen=True)
class ProfileOutput:
    name: str
    shape: tuple[int, ...]
    dtype: str
    engine_shape: tuple[int, ...] = ()


@dataclass(frozen=True)
class PreprocessProfile:
    color_format: str = ""
    scale: float | None = None
    offsets: tuple[float, ...] = ()
    mean: tuple[float, ...] = ()
    std: tuple[float, ...] = ()
    resize_mode: str = ""
    symmetric_padding: bool = False
    padding_value: float = 0.0

    @property
    def configured(self) -> bool:
        return bool(self.color_format and self.scale is not None and self.resize_mode)


@dataclass(frozen=True)
class DecoderProfile:
    parser_type: str = ""
    class_count: int = 0
    bbox_format: str = ""
    has_objectness: bool | None = None

    @property
    def configured(self) -> bool:
        return bool(
            self.parser_type
            and self.class_count > 0
            and self.bbox_format
            and self.has_objectness is not None
        )


@dataclass(frozen=True)
class PostprocessProfile:
    confidence_threshold: float = 0.25
    nms_threshold: float = 0.45
    max_detections: int = 300


@dataclass(frozen=True)
class ParserCandidate:
    parser_type: str
    confidence: str
    reason: str
    requires_confirmation: bool = True


@dataclass(frozen=True)
class ProfileValidation:
    status: str = "not_run"
    validated_at: str = ""
    engine_execution_ok: bool = False
    decoder_ok: bool = False
    nms_ok: bool = False
    detection_batch_ok: bool = False
    profile_fingerprint: str = ""
    issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelProfile:
    schema_version: int
    model_id: str
    display_name: str
    status: ModelStatus
    engine: EngineIdentity
    inspection: EngineInspectionResult
    input: ProfileInput
    outputs: tuple[ProfileOutput, ...]
    preprocess: PreprocessProfile = field(default_factory=PreprocessProfile)
    decoder: DecoderProfile = field(default_factory=DecoderProfile)
    postprocess: PostprocessProfile = field(default_factory=PostprocessProfile)
    labels: tuple[str, ...] = ()
    parser_candidates: tuple[ParserCandidate, ...] = ()
    validation: ProfileValidation = field(default_factory=ProfileValidation)


class ModelProfileResolver:
    def resolve(
        self,
        *,
        engine_path: Path,
        inspection: EngineInspectionResult,
        display_name: str,
    ) -> ModelProfile:
        path = Path(engine_path)
        stat = path.stat()
        identity = EngineIdentity(
            path=str(path.expanduser().resolve(strict=False)),
            sha256=f"sha256:{sha256_file(path)}",
            file_size=int(stat.st_size),
            modified_at_ns=int(stat.st_mtime_ns),
        )
        if inspection.inputs:
            engine_input = inspection.inputs[0]
            runtime_shape, profile_index = _runtime_shape(inspection, engine_input)
            profile_input = ProfileInput(
                name=engine_input.name,
                runtime_shape=runtime_shape,
                engine_shape=engine_input.engine_shape,
                dtype=engine_input.data_type,
                profile_index=profile_index,
            )
        else:
            profile_input = ProfileInput(
                name="",
                runtime_shape=(),
                engine_shape=(),
                dtype="",
            )
        outputs = tuple(
            ProfileOutput(
                name=output.name,
                shape=_runtime_output_shape(output.engine_shape),
                dtype=output.data_type,
                engine_shape=output.engine_shape,
            )
            for output in inspection.outputs
        )
        status = (
            ModelStatus.NEEDS_CONFIGURATION
            if inspection.deserialize_ok and inspection.compatible
            else ModelStatus.INCOMPATIBLE
        )
        return ModelProfile(
            schema_version=MODEL_PROFILE_SCHEMA_VERSION,
            model_id=identity.sha256,
            display_name=str(display_name).strip() or path.stem,
            status=status,
            engine=identity,
            inspection=inspection,
            input=profile_input,
            outputs=outputs,
            parser_candidates=_parser_candidates(inspection.outputs),
        )

    def configure(
        self,
        profile: ModelProfile,
        *,
        color_format: str,
        scale: float,
        resize_mode: str,
        parser_type: str,
        class_count: int,
        labels: list[str] | tuple[str, ...],
        bbox_format: str,
        has_objectness: bool,
        offsets: list[float] | tuple[float, ...] = (),
        mean: list[float] | tuple[float, ...] = (),
        std: list[float] | tuple[float, ...] = (),
        symmetric_padding: bool = False,
        padding_value: float = 0.0,
    ) -> ModelProfile:
        if not profile.inspection.deserialize_ok or not profile.inspection.compatible:
            raise ValueError("an incompatible TensorRT engine cannot be configured")
        color = str(color_format).strip().upper()
        if color not in {"RGB", "BGR"}:
            raise ValueError("preprocess color_format must be RGB or BGR")
        scale_value = float(scale)
        if not isfinite(scale_value) or scale_value <= 0.0:
            raise ValueError("preprocess scale must be finite and positive")
        resize = str(resize_mode).strip().lower()
        if resize not in {"direct", "letterbox"}:
            raise ValueError("preprocess resize_mode must be direct or letterbox")
        parser = get_parser_definition(parser_type)
        count = int(class_count)
        if count <= 0:
            raise ValueError("decoder class_count must be positive")
        normalized_labels = tuple(str(label).strip() for label in labels)
        if len(normalized_labels) != count or any(not label for label in normalized_labels):
            raise ValueError("labels must contain exactly class_count non-empty values")
        bbox = str(bbox_format).strip().lower()
        if bbox not in {"xywh", "xyxy"}:
            raise ValueError("decoder bbox_format must be xywh or xyxy")
        if parser.parser_type in {"yolov5_raw", "yolov8_raw", "yolo11_raw"}:
            if bbox != "xywh":
                raise ValueError(f"{parser.parser_type} requires bbox_format=xywh")
            expected_objectness = parser.parser_type == "yolov5_raw"
            if bool(has_objectness) is not expected_objectness:
                raise ValueError(
                    f"{parser.parser_type} objectness contract requires "
                    f"has_objectness={str(expected_objectness).lower()}"
                )
        normalized_offsets = _channel_sequence(offsets, "offsets", allow_zero=True)
        normalized_mean = _channel_sequence(mean, "mean", allow_zero=True)
        normalized_std = _channel_sequence(std, "std", allow_zero=False)
        padding = float(padding_value)
        if not isfinite(padding):
            raise ValueError("preprocess padding_value must be finite")
        _validate_parser_output_contract(
            profile,
            parser_type=parser.parser_type,
            class_count=count,
            has_objectness=bool(has_objectness),
        )
        return replace(
            profile,
            status=ModelStatus.READY_FOR_PROBE,
            preprocess=PreprocessProfile(
                color_format=color,
                scale=scale_value,
                offsets=normalized_offsets,
                mean=normalized_mean,
                std=normalized_std,
                resize_mode=resize,
                symmetric_padding=bool(symmetric_padding),
                padding_value=padding,
            ),
            decoder=DecoderProfile(
                parser_type=parser.parser_type,
                class_count=count,
                bbox_format=bbox,
                has_objectness=bool(has_objectness),
            ),
            labels=normalized_labels,
            validation=ProfileValidation(),
        )


def model_profile_validation_fingerprint(profile: ModelProfile) -> str:
    payload = {
        "schema_version": profile.schema_version,
        "model_id": profile.model_id,
        "engine_sha256": profile.engine.sha256,
        "input": asdict(profile.input),
        "outputs": [asdict(output) for output in profile.outputs],
        "preprocess": asdict(profile.preprocess),
        "decoder": asdict(profile.decoder),
        "labels": list(profile.labels),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _channel_sequence(
    values: list[float] | tuple[float, ...],
    name: str,
    *,
    allow_zero: bool,
) -> tuple[float, ...]:
    normalized = tuple(float(value) for value in values)
    if len(normalized) not in {0, 1, 3}:
        raise ValueError(f"preprocess {name} must contain one or three values")
    if any(not isfinite(value) for value in normalized):
        raise ValueError(f"preprocess {name} values must be finite")
    if not allow_zero and any(value == 0.0 for value in normalized):
        raise ValueError("preprocess std values must be non-zero")
    return normalized


def _runtime_shape(
    inspection: EngineInspectionResult,
    input_tensor: TensorDescriptor,
) -> tuple[tuple[int, ...], int]:
    if input_tensor.engine_shape and all(value > 0 for value in input_tensor.engine_shape):
        return input_tensor.engine_shape, 0
    for profile in inspection.profiles:
        shape_range = profile.input_ranges.get(input_tensor.name)
        if shape_range is not None and all(value > 0 for value in shape_range.optimum):
            return shape_range.optimum, profile.profile_index
    return (), 0


def _runtime_output_shape(engine_shape: tuple[int, ...]) -> tuple[int, ...]:
    if not engine_shape:
        return ()
    if engine_shape[0] < 0 and all(value > 0 for value in engine_shape[1:]):
        return (1, *engine_shape[1:])
    return engine_shape


def _parser_candidates(
    outputs: tuple[TensorDescriptor, ...],
) -> tuple[ParserCandidate, ...]:
    if len(outputs) != 1:
        return ()
    output = outputs[0]
    dimensions = [value for value in output.engine_shape if value > 0]
    if len(dimensions) >= 2:
        first, second = dimensions[-2:]
        columns = min(first, second)
        candidates = max(first, second)
        if 5 <= columns <= 512 and candidates > columns:
            name = output.name.lower()
            if "v5" in name:
                parser_type = "yolov5_raw"
                confidence = "high"
            elif "11" in name or "yolo11" in name:
                parser_type = "yolo11_raw"
                confidence = "high"
            else:
                parser_type = "yolov8_raw"
                confidence = "medium"
            return (
                ParserCandidate(
                    parser_type=parser_type,
                    confidence=confidence,
                    reason=(
                        "single detection output has a supported candidates-by-channels shape; "
                        "objectness and class semantics still require confirmation"
                    ),
                ),
            )
    return ()


def _validate_parser_output_contract(
    profile: ModelProfile,
    *,
    parser_type: str,
    class_count: int,
    has_objectness: bool,
) -> None:
    if len(profile.outputs) != 1:
        raise ValueError(f"{parser_type} requires exactly one raw detection output")
    dimensions = [dimension for dimension in profile.outputs[0].shape if dimension > 0]
    if len(dimensions) < 2:
        raise ValueError("raw YOLO output shape is unresolved")
    expected_columns = (5 if has_objectness else 4) + class_count
    if expected_columns not in dimensions[-2:]:
        raise ValueError(
            "decoder class/objectness contract does not match TensorRT output shape "
            f"(expected columns={expected_columns}, shape={profile.outputs[0].shape})"
        )


__all__ = [
    "DecoderProfile",
    "EngineIdentity",
    "MODEL_PROFILE_SCHEMA_VERSION",
    "ModelProfile",
    "ModelProfileResolver",
    "ModelStatus",
    "ParserCandidate",
    "PostprocessProfile",
    "PreprocessProfile",
    "ProfileInput",
    "ProfileOutput",
    "ProfileValidation",
    "model_profile_validation_fingerprint",
]
