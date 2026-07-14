from __future__ import annotations

from dataclasses import asdict, replace
from enum import Enum
import json
from pathlib import Path
import tempfile
from typing import Any

from novasight.model_registry.fingerprint import sha256_file
from novasight.model_registry.manifest import ModelManifest

from .contracts import (
    EngineInspectionErrorCode,
    EngineInspectionResult,
    OptimizationProfileDescriptor,
    ShapeRange,
    TensorDescriptor,
)
from .profile import (
    DecoderProfile,
    EngineIdentity,
    ModelProfile,
    ModelStatus,
    ParserCandidate,
    PostprocessProfile,
    PreprocessProfile,
    ProfileInput,
    ProfileOutput,
    ProfileValidation,
    MODEL_PROFILE_SCHEMA_VERSION,
    model_profile_validation_fingerprint,
)


class ModelProfileStore:
    def path_for_engine(self, engine_path: Path) -> Path:
        path = Path(engine_path)
        return path.with_name(f"{path.name}.manifest.json")

    def legacy_path_for_engine(self, engine_path: Path) -> Path:
        path = Path(engine_path)
        return path.with_name(f"{path.name}.profile.json")

    def existing_path_for_engine(self, engine_path: Path) -> Path:
        current = self.path_for_engine(engine_path)
        if current.is_file():
            return current
        legacy = self.legacy_path_for_engine(engine_path)
        return legacy if legacy.is_file() else current

    def contains_model_profile(self, path: Path) -> bool:
        candidate = Path(path)
        if not candidate.is_file():
            return False
        if candidate.name.endswith(".profile.json"):
            return True
        try:
            raw = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return isinstance(raw, dict) and isinstance(raw.get("model_profile"), dict)

    def write(
        self,
        profile: ModelProfile,
        path: Path | None = None,
        *,
        runtime_manifest: ModelManifest | None = None,
    ) -> Path:
        target = Path(path) if path is not None else self.path_for_engine(Path(profile.engine.path))
        target.parent.mkdir(parents=True, exist_ok=True)
        existing: dict[str, Any] = {}
        if target.is_file():
            try:
                loaded = json.loads(target.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                loaded = {}
            if isinstance(loaded, dict):
                existing = loaded
        if runtime_manifest is not None:
            payload = runtime_manifest.to_dict()
        elif profile.status in {ModelStatus.VALIDATED, ModelStatus.ACTIVE}:
            payload = {
                key: value
                for key, value in existing.items()
                if key != "model_profile"
            }
        else:
            payload = {}
        payload.setdefault("schema_version", MODEL_PROFILE_SCHEMA_VERSION)
        payload["manifest_kind"] = "novasight_model"
        payload["model_profile"] = _json_value(asdict(profile))
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                handle.write(text)
                handle.flush()
                temporary = Path(handle.name)
            temporary.replace(target)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return target

    def load(self, path: Path) -> ModelProfile:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("model manifest must contain a JSON object")
        if "model_profile" in raw and int(raw.get("schema_version", 0)) != MODEL_PROFILE_SCHEMA_VERSION:
            raise ValueError(
                "unsupported unified model manifest schema_version "
                f"{raw.get('schema_version')}; expected {MODEL_PROFILE_SCHEMA_VERSION}"
            )
        profile_raw = raw.get("model_profile", raw)
        if not isinstance(profile_raw, dict):
            raise ValueError("model manifest model_profile must contain a JSON object")
        profile = _profile_from_dict(profile_raw)
        engine_path = Path(profile.engine.path)
        try:
            content_matches = (
                engine_path.is_file()
                and engine_path.stat().st_size == profile.engine.file_size
                and f"sha256:{sha256_file(engine_path)}" == profile.engine.sha256
            )
        except OSError:
            content_matches = False
        if content_matches:
            if (
                profile.status in {ModelStatus.VALIDATED, ModelStatus.ACTIVE}
                and profile.validation.profile_fingerprint
                != model_profile_validation_fingerprint(profile)
            ):
                return replace(
                    profile,
                    status=ModelStatus.INVALID,
                    validation=replace(
                        profile.validation,
                        status="invalid",
                        issues=(
                            *profile.validation.issues,
                            "PROFILE_SEMANTICS_CHANGED",
                        ),
                    ),
                )
            return profile
        return replace(
            profile,
            status=ModelStatus.UNINSPECTED,
            validation=ProfileValidation(issues=("ENGINE_CONTENT_CHANGED",)),
        )


def _profile_from_dict(raw: dict[str, Any]) -> ModelProfile:
    schema_version = int(raw.get("schema_version", 0))
    if schema_version != MODEL_PROFILE_SCHEMA_VERSION:
        raise ValueError(
            "unsupported ModelProfile schema_version "
            f"{schema_version}; expected {MODEL_PROFILE_SCHEMA_VERSION}"
        )
    inspection_raw = dict(raw["inspection"])
    inputs = tuple(_tensor_from_dict(item) for item in inspection_raw.get("inputs", []))
    outputs = tuple(_tensor_from_dict(item) for item in inspection_raw.get("outputs", []))
    profiles = tuple(
        OptimizationProfileDescriptor(
            profile_index=int(item["profile_index"]),
            input_ranges={
                str(name): ShapeRange(
                    minimum=tuple(int(value) for value in shape_range["minimum"]),
                    optimum=tuple(int(value) for value in shape_range["optimum"]),
                    maximum=tuple(int(value) for value in shape_range["maximum"]),
                )
                for name, shape_range in dict(item.get("input_ranges", {})).items()
            },
        )
        for item in inspection_raw.get("profiles", [])
    )
    error_code_raw = inspection_raw.get("error_code")
    inspection = EngineInspectionResult(
        deserialize_ok=bool(inspection_raw.get("deserialize_ok")),
        compatible=bool(inspection_raw.get("compatible")),
        engine_name=str(inspection_raw.get("engine_name", "")),
        inputs=inputs,
        outputs=outputs,
        profiles=profiles,
        has_dynamic_shape=bool(inspection_raw.get("has_dynamic_shape")),
        has_shape_input=bool(inspection_raw.get("has_shape_input")),
        requires_plugin=bool(inspection_raw.get("requires_plugin")),
        error_code=(
            EngineInspectionErrorCode(str(error_code_raw)) if error_code_raw else None
        ),
        raw_error=str(inspection_raw.get("raw_error", "")),
        warnings=tuple(str(item) for item in inspection_raw.get("warnings", [])),
    )
    engine_raw = dict(raw["engine"])
    input_raw = dict(raw["input"])
    preprocess_raw = dict(raw.get("preprocess", {}))
    decoder_raw = dict(raw.get("decoder", {}))
    postprocess_raw = dict(raw.get("postprocess", {}))
    validation_raw = dict(raw.get("validation", {}))
    return ModelProfile(
        schema_version=schema_version,
        model_id=str(raw["model_id"]),
        display_name=str(raw["display_name"]),
        status=ModelStatus(str(raw["status"])),
        engine=EngineIdentity(
            path=str(engine_raw["path"]),
            sha256=str(engine_raw["sha256"]),
            file_size=int(engine_raw["file_size"]),
            modified_at_ns=int(engine_raw["modified_at_ns"]),
        ),
        inspection=inspection,
        input=ProfileInput(
            name=str(input_raw["name"]),
            runtime_shape=tuple(int(value) for value in input_raw["runtime_shape"]),
            engine_shape=tuple(int(value) for value in input_raw["engine_shape"]),
            dtype=str(input_raw["dtype"]),
            layout=str(input_raw.get("layout", "NCHW")),
            profile_index=int(input_raw.get("profile_index", 0)),
        ),
        outputs=tuple(
            ProfileOutput(
                name=str(item["name"]),
                shape=tuple(int(value) for value in item["shape"]),
                dtype=str(item["dtype"]),
                engine_shape=tuple(
                    int(value) for value in item.get("engine_shape", item["shape"])
                ),
            )
            for item in raw.get("outputs", [])
        ),
        preprocess=PreprocessProfile(
            color_format=str(preprocess_raw.get("color_format", "")),
            scale=(
                float(preprocess_raw["scale"])
                if preprocess_raw.get("scale") is not None
                else None
            ),
            offsets=tuple(float(value) for value in preprocess_raw.get("offsets", [])),
            mean=tuple(float(value) for value in preprocess_raw.get("mean", [])),
            std=tuple(float(value) for value in preprocess_raw.get("std", [])),
            resize_mode=str(preprocess_raw.get("resize_mode", "")),
            symmetric_padding=bool(preprocess_raw.get("symmetric_padding", False)),
            padding_value=float(preprocess_raw.get("padding_value", 0.0)),
        ),
        decoder=DecoderProfile(
            parser_type=str(decoder_raw.get("parser_type", "")),
            class_count=int(decoder_raw.get("class_count", 0)),
            bbox_format=str(decoder_raw.get("bbox_format", "")),
            has_objectness=decoder_raw.get("has_objectness"),
        ),
        postprocess=PostprocessProfile(
            confidence_threshold=float(postprocess_raw.get("confidence_threshold", 0.25)),
            nms_threshold=float(postprocess_raw.get("nms_threshold", 0.45)),
            max_detections=int(postprocess_raw.get("max_detections", 300)),
        ),
        labels=tuple(str(value) for value in raw.get("labels", [])),
        parser_candidates=tuple(
            ParserCandidate(
                parser_type=str(item["parser_type"]),
                confidence=str(item["confidence"]),
                reason=str(item["reason"]),
                requires_confirmation=bool(item.get("requires_confirmation", True)),
            )
            for item in raw.get("parser_candidates", [])
        ),
        validation=ProfileValidation(
            status=str(validation_raw.get("status", "not_run")),
            validated_at=str(validation_raw.get("validated_at", "")),
            engine_execution_ok=bool(validation_raw.get("engine_execution_ok", False)),
            decoder_ok=bool(validation_raw.get("decoder_ok", False)),
            nms_ok=bool(validation_raw.get("nms_ok", False)),
            detection_batch_ok=bool(validation_raw.get("detection_batch_ok", False)),
            profile_fingerprint=str(validation_raw.get("profile_fingerprint", "")),
            issues=tuple(str(item) for item in validation_raw.get("issues", [])),
        ),
    )


def _tensor_from_dict(raw: dict[str, Any]) -> TensorDescriptor:
    return TensorDescriptor(
        name=str(raw["name"]),
        io_mode=str(raw["io_mode"]),
        engine_shape=tuple(int(value) for value in raw["engine_shape"]),
        data_type=str(raw["data_type"]),
        tensor_format=str(raw["tensor_format"]),
        is_shape_tensor=bool(raw["is_shape_tensor"]),
        bytes_per_component=int(raw["bytes_per_component"]),
        components_per_element=int(raw["components_per_element"]),
        vectorized_dim=int(raw["vectorized_dim"]),
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


__all__ = ["ModelProfileStore"]
