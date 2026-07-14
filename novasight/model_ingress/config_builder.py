from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from novasight.deepstream.nvinfer_config import generate_nvinfer_config
from novasight.model_registry.fingerprint import sha256_file
from novasight.model_registry.manifest import (
    ModelManifest,
    TensorSpec,
    build_engine_manifest,
    compute_model_fingerprint,
)

from .parser_registry import get_parser_definition
from .profile import (
    ModelProfile,
    ModelStatus,
    PostprocessProfile,
    PreprocessProfile,
    model_profile_validation_fingerprint,
)


@dataclass(frozen=True)
class RuntimeDecoderConfig:
    parser_type: str
    runtime_decoder: str
    class_count: int
    bbox_format: str
    has_objectness: bool
    labels: tuple[str, ...]


@dataclass(frozen=True)
class RuntimeInferenceConfig:
    model_id: str
    model_fingerprint: str
    engine_path: Path
    engine_sha256: str
    input_name: str
    input_shape: tuple[int, ...]
    input_dtype: str
    output_names: tuple[str, ...]
    output_shapes: tuple[tuple[int, ...], ...]
    output_dtypes: tuple[str, ...]
    preprocess: PreprocessProfile
    decoder: RuntimeDecoderConfig
    postprocess: PostprocessProfile
    manifest: ModelManifest
    deepstream_nvinfer: str


class InferenceConfigBuilder:
    def build(
        self,
        profile: ModelProfile,
        *,
        parser_library_path: Path,
    ) -> RuntimeInferenceConfig:
        self._validate_profile(profile)
        parser = get_parser_definition(profile.decoder.parser_type)
        if parser.built_in_nms:
            raise ValueError(
                "the current DeepStream native parser does not support built-in NMS outputs"
            )
        if len(profile.outputs) != 1:
            raise ValueError("the current DeepStream runtime requires exactly one output tensor")
        if profile.preprocess.offsets or profile.preprocess.mean or profile.preprocess.std:
            raise ValueError(
                "the current DeepStream nvinfer path cannot represent ModelProfile "
                "mean/std/offsets exactly; use a supported CUDA preprocess contract"
            )
        output = profile.outputs[0]
        if not profile.input.runtime_shape or any(value <= 0 for value in profile.input.runtime_shape):
            raise ValueError("ModelProfile input runtime_shape must be concrete")
        if not output.shape or any(value <= 0 for value in output.shape):
            raise ValueError("ModelProfile output runtime shape must be concrete")
        engine_path = Path(profile.engine.path)
        manifest = build_engine_manifest(
            model_id=profile.model_id,
            display_name=profile.display_name,
            engine_path=engine_path,
            input_spec=TensorSpec(
                name=profile.input.name,
                shape=list(profile.input.runtime_shape),
                dtype=profile.input.dtype,
                layout=profile.input.layout,
            ),
            output_spec=TensorSpec(
                name=output.name,
                shape=list(output.shape),
                dtype=output.dtype,
                layout="NCHW",
            ),
            class_count=profile.decoder.class_count,
            class_names=list(profile.labels),
            confidence_threshold=profile.postprocess.confidence_threshold,
            nms_iou_threshold=profile.postprocess.nms_threshold,
            runtime_precision=_runtime_precision(profile.input.dtype),
            input_color_format=profile.preprocess.color_format,
            input_scale_factor=float(profile.preprocess.scale),
            maintain_aspect_ratio=profile.preprocess.resize_mode == "letterbox",
            symmetric_padding=profile.preprocess.symmetric_padding,
            output_has_objectness=bool(profile.decoder.has_objectness),
            validated=True,
        )
        manifest = replace(
            manifest,
            postprocess=replace(
                manifest.postprocess,
                max_detections=profile.postprocess.max_detections,
            ),
        )
        manifest = replace(
            manifest,
            model_fingerprint=compute_model_fingerprint(manifest),
        )
        nvinfer = generate_nvinfer_config(
            manifest,
            engine_path=engine_path,
            parser_library_path=parser_library_path,
        )
        return RuntimeInferenceConfig(
            model_id=profile.model_id,
            model_fingerprint=manifest.model_fingerprint,
            engine_path=engine_path,
            engine_sha256=profile.engine.sha256,
            input_name=profile.input.name,
            input_shape=profile.input.runtime_shape,
            input_dtype=profile.input.dtype,
            output_names=tuple(item.name for item in profile.outputs),
            output_shapes=tuple(item.shape for item in profile.outputs),
            output_dtypes=tuple(item.dtype for item in profile.outputs),
            preprocess=profile.preprocess,
            decoder=RuntimeDecoderConfig(
                parser_type=parser.parser_type,
                runtime_decoder=parser.runtime_decoder,
                class_count=profile.decoder.class_count,
                bbox_format=profile.decoder.bbox_format,
                has_objectness=bool(profile.decoder.has_objectness),
                labels=profile.labels,
            ),
            postprocess=profile.postprocess,
            manifest=manifest,
            deepstream_nvinfer=nvinfer,
        )

    @staticmethod
    def _validate_profile(profile: ModelProfile) -> None:
        if profile.status not in {ModelStatus.VALIDATED, ModelStatus.ACTIVE}:
            raise ValueError("runtime config can only be built from a validated ModelProfile")
        validation = profile.validation
        if not (
            validation.engine_execution_ok
            and validation.decoder_ok
            and validation.nms_ok
            and validation.detection_batch_ok
        ):
            raise ValueError("ModelProfile validation report does not allow runtime activation")
        if validation.profile_fingerprint != model_profile_validation_fingerprint(profile):
            raise ValueError(
                "ModelProfile semantics changed after validation; run diagnostics again"
            )
        engine_path = Path(profile.engine.path)
        if not engine_path.is_file():
            raise ValueError(f"ModelProfile engine file is missing: {engine_path}")
        if engine_path.stat().st_size != profile.engine.file_size:
            raise ValueError("ModelProfile engine size no longer matches the validated artifact")
        if f"sha256:{sha256_file(engine_path)}" != profile.engine.sha256:
            raise ValueError("ModelProfile engine checksum no longer matches the validated artifact")


def _runtime_precision(dtype: str) -> str:
    normalized = str(dtype).strip().lower()
    if normalized in {"float16", "fp16", "half"}:
        return "fp16"
    if normalized in {"float32", "fp32", "float"}:
        return "fp32"
    if normalized == "int8":
        return "int8"
    raise ValueError(f"unsupported runtime input dtype: {dtype}")


__all__ = [
    "InferenceConfigBuilder",
    "RuntimeDecoderConfig",
    "RuntimeInferenceConfig",
]
