from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .fingerprint import sha256_file
from .manifest import ModelManifest, read_manifest


ModelScanStatus = Literal["ready", "need_confirm", "invalid", "unsupported"]
SUPPORTED_MODEL_SUFFIXES = {".engine", ".onnx"}


@dataclass(frozen=True)
class ModelArtifactScanResult:
    path: Path
    kind: str
    status: ModelScanStatus
    reason: str
    sha256: str
    size_bytes: int
    manifest_path: Path | None = None
    deepstream_config_path: Path | None = None
    model_fingerprint: str = ""
    manifest: ModelManifest | None = None


def scan_model_artifacts(root: Path) -> list[ModelArtifactScanResult]:
    root = Path(root)
    if not root.exists():
        return []
    results = [
        inspect_model_artifact(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_MODEL_SUFFIXES
    ]
    return results


def inspect_model_artifact(path: Path) -> ModelArtifactScanResult:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_MODEL_SUFFIXES:
        return ModelArtifactScanResult(
            path=path,
            kind=suffix.lstrip(".") or "unknown",
            status="unsupported",
            reason="model artifact suffix is not supported",
            sha256="",
            size_bytes=0,
        )
    artifact_sha256 = sha256_file(path)
    size_bytes = path.stat().st_size
    manifest_path = path.with_name("model.manifest.json")
    deepstream_config_path = path.with_name("deepstream.ini")
    if not manifest_path.exists():
        return ModelArtifactScanResult(
            path=path,
            kind=suffix.lstrip("."),
            status="need_confirm",
            reason="model manifest is missing",
            sha256=artifact_sha256,
            size_bytes=size_bytes,
            manifest_path=manifest_path,
            deepstream_config_path=deepstream_config_path,
        )
    try:
        manifest = read_manifest(manifest_path)
    except Exception as exc:
        return ModelArtifactScanResult(
            path=path,
            kind=suffix.lstrip("."),
            status="invalid",
            reason=f"model manifest is invalid: {exc}",
            sha256=artifact_sha256,
            size_bytes=size_bytes,
            manifest_path=manifest_path,
            deepstream_config_path=deepstream_config_path,
        )
    if manifest.artifact.sha256 != artifact_sha256:
        return ModelArtifactScanResult(
            path=path,
            kind=suffix.lstrip("."),
            status="invalid",
            reason="model manifest artifact sha256 does not match file",
            sha256=artifact_sha256,
            size_bytes=size_bytes,
            manifest_path=manifest_path,
            deepstream_config_path=deepstream_config_path,
            model_fingerprint=manifest.model_fingerprint,
            manifest=manifest,
        )
    if manifest.artifact.size_bytes != size_bytes:
        return ModelArtifactScanResult(
            path=path,
            kind=suffix.lstrip("."),
            status="invalid",
            reason="model manifest artifact size does not match file",
            sha256=artifact_sha256,
            size_bytes=size_bytes,
            manifest_path=manifest_path,
            deepstream_config_path=deepstream_config_path,
            model_fingerprint=manifest.model_fingerprint,
            manifest=manifest,
        )
    return ModelArtifactScanResult(
        path=path,
        kind=suffix.lstrip("."),
        status="ready",
        reason="model manifest matches artifact",
        sha256=artifact_sha256,
        size_bytes=size_bytes,
        manifest_path=manifest_path,
        deepstream_config_path=deepstream_config_path,
        model_fingerprint=manifest.model_fingerprint,
        manifest=manifest,
    )
