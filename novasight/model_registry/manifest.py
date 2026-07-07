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
    backend: str = "deepstream"
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
class DeepStreamSpec:
    network_type: int = 100
    output_tensor_meta: bool = True
    gie_unique_id: int = 1
    interval: int = 0


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
    deepstream: DeepStreamSpec = field(default_factory=DeepStreamSpec)
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
    confidence_threshold: float = 0.25,
    nms_iou_threshold: float = 0.45,
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
        runtime=RuntimeInfo(),
        input=InputSpec(
            name=_require_non_empty(input_spec.name, "input.name"),
            shape=_validate_shape(input_spec.shape, "input.shape"),
            dtype=_require_non_empty(input_spec.dtype, "input.dtype"),
            layout=_require_non_empty(input_spec.layout, "input.layout"),
        ),
        output=OutputSpec(
            name=_require_non_empty(output_spec.name, "output.name"),
            shape=_validate_shape(output_spec.shape, "output.shape"),
            dtype=_require_non_empty(output_spec.dtype, "output.dtype"),
            layout=_require_non_empty(output_spec.layout, "output.layout"),
            class_count=max(0, int(class_count)),
        ),
        postprocess=PostprocessSpec(
            confidence_threshold=float(confidence_threshold),
            nms_iou_threshold=float(nms_iou_threshold),
        ),
        deepstream=DeepStreamSpec(),
        validated=bool(validated),
    )
    return replace(manifest, model_fingerprint=compute_model_fingerprint(manifest))


def compute_model_fingerprint(manifest: ModelManifest) -> str:
    payload = {
        "artifact_sha256": manifest.artifact.sha256,
        "input": {
            "name": manifest.input.name,
            "shape": list(manifest.input.shape),
            "dtype": manifest.input.dtype,
            "layout": manifest.input.layout,
        },
        "output": {
            "name": manifest.output.name,
            "shape": list(manifest.output.shape),
            "dtype": manifest.output.dtype,
            "layout": manifest.output.layout,
            "format": manifest.output.format,
            "class_count": manifest.output.class_count,
            "has_objectness": manifest.output.has_objectness,
            "coordinate_mode": manifest.output.coordinate_mode,
        },
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


def manifest_from_dict(raw: dict[str, Any]) -> ModelManifest:
    artifact = ArtifactInfo(**dict(raw["artifact"]))
    runtime = RuntimeInfo(**dict(raw.get("runtime", {})))
    input_spec = InputSpec(**dict(raw["input"]))
    output_spec = OutputSpec(**dict(raw["output"]))
    postprocess = PostprocessSpec(**dict(raw.get("postprocess", {})))
    deepstream = DeepStreamSpec(**dict(raw.get("deepstream", {})))
    manifest = ModelManifest(
        schema_version=int(raw.get("schema_version", MODEL_MANIFEST_SCHEMA_VERSION)),
        model_id=str(raw["model_id"]),
        display_name=str(raw["display_name"]),
        artifact=artifact,
        runtime=runtime,
        input=input_spec,
        output=output_spec,
        postprocess=postprocess,
        deepstream=deepstream,
        validated=bool(raw.get("validated", False)),
        model_fingerprint=str(raw.get("model_fingerprint", "")),
    )
    expected = compute_model_fingerprint(manifest)
    if manifest.model_fingerprint and manifest.model_fingerprint != expected:
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
