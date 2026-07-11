from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path
from typing import Any

from .fingerprint import sha256_file, sha256_text, stable_json


MODEL_MANIFEST_SCHEMA_VERSION = 1
MODEL_PARSER_SCHEMA_VERSION = "yolo-v1"


@dataclass(frozen=True)
class ArtifactInfo:
    engine_path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class RuntimeInfo:
    backend: str = "custom_tensorrt"
    precision: str = "fp16"
    batch_size: int = 1


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: list[int]
    dtype: str
    layout: str = "NCHW"


@dataclass(frozen=True)
class InputSpec(TensorSpec):
    color_format: str = "RGB"
    scale_factor: float = 1.0 / 255.0
    maintain_aspect_ratio: bool = False
    symmetric_padding: bool = False


@dataclass(frozen=True)
class OutputSpec(TensorSpec):
    format: str = "yolo_cxcywh_class_scores"
    class_count: int = 0
    class_names: list[str] = field(default_factory=list)
    has_objectness: bool = False
    scores_are_sigmoid: bool = True
    coordinate_mode: str = "pixel"


@dataclass(frozen=True)
class PostprocessSpec:
    parser: str = "yolo"
    confidence_threshold: float = 0.25
    nms_iou_threshold: float = 0.45
    class_aware_nms: bool = True
    max_detections: int = 300


@dataclass(frozen=True)
class ModelManifest:
    schema_version: int
    model_id: str
    display_name: str
    artifact: ArtifactInfo
    runtime: RuntimeInfo
    input: InputSpec
    output: OutputSpec
    postprocess: PostprocessSpec = field(default_factory=PostprocessSpec)
    validated: bool = False
    model_fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if not payload.get("model_fingerprint"):
            payload["model_fingerprint"] = compute_model_fingerprint(self)
        return payload


def build_engine_manifest(
    *,
    model_id: str,
    display_name: str,
    engine_path: Path,
    input_spec: TensorSpec,
    output_spec: TensorSpec,
    class_count: int,
    class_names: list[str] | None = None,
    confidence_threshold: float = 0.25,
    nms_iou_threshold: float = 0.45,
    runtime_precision: str = "fp16",
    input_color_format: str = "RGB",
    input_scale_factor: float = 1.0 / 255.0,
    maintain_aspect_ratio: bool = False,
    symmetric_padding: bool = False,
    output_format: str = "yolo_cxcywh_class_scores",
    output_has_objectness: bool = False,
    output_coordinate_mode: str = "pixel",
    postprocess_parser: str = "yolo",
    validated: bool = False,
) -> ModelManifest:
    engine_path = Path(engine_path)
    artifact = ArtifactInfo(
        engine_path=engine_path.name,
        sha256=sha256_file(engine_path),
        size_bytes=engine_path.stat().st_size,
    )
    manifest = ModelManifest(
        schema_version=MODEL_MANIFEST_SCHEMA_VERSION,
        model_id=_require_non_empty(model_id, "model_id"),
        display_name=_require_non_empty(display_name, "display_name"),
        artifact=artifact,
        runtime=RuntimeInfo(precision=_require_non_empty(runtime_precision, "runtime.precision")),
        input=InputSpec(
            name=_require_non_empty(input_spec.name, "input.name"),
            shape=_validate_shape(input_spec.shape, "input.shape"),
            dtype=_require_non_empty(input_spec.dtype, "input.dtype"),
            layout=_require_non_empty(input_spec.layout, "input.layout"),
            color_format=_require_non_empty(input_color_format, "input.color_format"),
            scale_factor=float(input_scale_factor),
            maintain_aspect_ratio=bool(maintain_aspect_ratio),
            symmetric_padding=bool(symmetric_padding),
        ),
        output=OutputSpec(
            name=_require_non_empty(output_spec.name, "output.name"),
            shape=_validate_shape(output_spec.shape, "output.shape"),
            dtype=_require_non_empty(output_spec.dtype, "output.dtype"),
            layout=_require_non_empty(output_spec.layout, "output.layout"),
            class_count=max(0, int(class_count)),
            class_names=_normalize_class_names(class_names, max(0, int(class_count))),
            format=_require_non_empty(output_format, "output.format"),
            has_objectness=bool(output_has_objectness),
            coordinate_mode=_require_non_empty(
                output_coordinate_mode,
                "output.coordinate_mode",
            ),
        ),
        postprocess=PostprocessSpec(
            parser=_require_non_empty(postprocess_parser, "postprocess.parser"),
            confidence_threshold=float(confidence_threshold),
            nms_iou_threshold=float(nms_iou_threshold),
        ),
        validated=bool(validated),
    )
    return replace(manifest, model_fingerprint=compute_model_fingerprint(manifest))


def compute_model_fingerprint(manifest: ModelManifest) -> str:
    return _compute_model_fingerprint(manifest, include_class_names=True)


def _compute_model_fingerprint(
    manifest: ModelManifest,
    *,
    include_class_names: bool,
) -> str:
    output_payload = {
        "name": manifest.output.name,
        "shape": list(manifest.output.shape),
        "dtype": manifest.output.dtype,
        "layout": manifest.output.layout,
        "format": manifest.output.format,
        "class_count": manifest.output.class_count,
        "has_objectness": manifest.output.has_objectness,
        "coordinate_mode": manifest.output.coordinate_mode,
    }
    if include_class_names:
        output_payload["class_names"] = list(manifest.output.class_names)
    payload = {
        "artifact_sha256": manifest.artifact.sha256,
        "input": {
            "name": manifest.input.name,
            "shape": list(manifest.input.shape),
            "dtype": manifest.input.dtype,
            "layout": manifest.input.layout,
        },
        "output": output_payload,
        "parser_schema": MODEL_PARSER_SCHEMA_VERSION,
    }
    return sha256_text(stable_json(payload))


def write_manifest(manifest: ModelManifest, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_manifest(path: Path) -> ModelManifest:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return manifest_from_dict(raw)


def validate_manifest_engine_artifact(manifest: ModelManifest, engine_path: Path) -> None:
    engine_path = Path(engine_path)
    if manifest.artifact.engine_path != engine_path.name:
        raise ValueError("model manifest artifact does not match engine file name")
    if not engine_path.is_file():
        raise ValueError(f"model engine file is missing: {engine_path}")
    actual_size = engine_path.stat().st_size
    if int(manifest.artifact.size_bytes) != int(actual_size):
        raise ValueError(
            "model manifest artifact size does not match engine file "
            f"(manifest={manifest.artifact.size_bytes}, actual={actual_size})"
        )
    actual_sha256 = sha256_file(engine_path)
    if manifest.artifact.sha256 != actual_sha256:
        raise ValueError("model manifest artifact checksum does not match engine file")


def manifest_from_dict(raw: dict[str, Any]) -> ModelManifest:
    raw_output = dict(raw["output"])
    artifact = ArtifactInfo(**dict(raw["artifact"]))
    runtime = RuntimeInfo(**dict(raw.get("runtime", {})))
    input_spec = InputSpec(**dict(raw["input"]))
    output_spec = OutputSpec(**raw_output)
    postprocess = PostprocessSpec(**dict(raw.get("postprocess", {})))
    manifest = ModelManifest(
        schema_version=int(raw.get("schema_version", MODEL_MANIFEST_SCHEMA_VERSION)),
        model_id=str(raw["model_id"]),
        display_name=str(raw["display_name"]),
        artifact=artifact,
        runtime=runtime,
        input=input_spec,
        output=output_spec,
        postprocess=postprocess,
        validated=bool(raw.get("validated", False)),
        model_fingerprint=str(raw.get("model_fingerprint", "")),
    )
    expected = compute_model_fingerprint(manifest)
    if manifest.model_fingerprint and manifest.model_fingerprint != expected:
        legacy_expected = ""
        if "class_names" not in raw_output:
            legacy_expected = _compute_model_fingerprint(manifest, include_class_names=False)
        if manifest.model_fingerprint != legacy_expected:
            raise ValueError("model manifest fingerprint does not match manifest content")
    if not manifest.model_fingerprint:
        return replace(manifest, model_fingerprint=expected)
    return manifest


def _require_non_empty(value: str, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} must not be empty")
    return text


def _validate_shape(value: list[int], label: str) -> list[int]:
    shape = [int(item) for item in value]
    if not shape or any(item <= 0 for item in shape):
        raise ValueError(f"{label} must contain positive dimensions")
    return shape


def _normalize_class_names(value: list[str] | None, class_count: int) -> list[str]:
    count = max(0, int(class_count))
    names = [str(item).strip() for item in (value or []) if str(item).strip()]
    normalized = names[:count]
    while len(normalized) < count:
        normalized.append(str(len(normalized)))
    return normalized
