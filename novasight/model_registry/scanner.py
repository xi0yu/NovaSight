from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .fingerprint import sha256_file
from .manifest import ModelManifest, read_manifest


ModelScanStatus = Literal["ready", "need_confirm", "invalid", "unsupported"]
SUPPORTED_MODEL_SUFFIXES = {".engine", ".onnx"}
_MODEL_SCAN_LOCK = threading.RLock()
_MODEL_SCAN_CACHE: dict[str, tuple[tuple[int, ...], "ModelArtifactScanResult"]] = {}


@dataclass(frozen=True)
class ModelArtifactScanResult:
    path: Path
    kind: str
    status: ModelScanStatus
    reason: str
    sha256: str
    size_bytes: int
    manifest_path: Path | None = None
    model_fingerprint: str = ""
    manifest: ModelManifest | None = None


def scan_model_artifacts(
    root: Path,
    *,
    force: bool = False,
) -> list[ModelArtifactScanResult]:
    root = Path(root)
    if not root.exists():
        return []
    results = [
        inspect_model_artifact(path, force=force)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_MODEL_SUFFIXES
    ]
    return results


def inspect_model_artifact(
    path: Path,
    *,
    force: bool = False,
) -> ModelArtifactScanResult:
    path = Path(path)
    cache_key = str(path.resolve(strict=False))
    signature = _artifact_signature(path)
    if not force:
        with _MODEL_SCAN_LOCK:
            cached = _MODEL_SCAN_CACHE.get(cache_key)
            if cached is not None and cached[0] == signature:
                return cached[1]
    result = _inspect_model_artifact_uncached(path)
    with _MODEL_SCAN_LOCK:
        _MODEL_SCAN_CACHE[cache_key] = (signature, result)
    return result


def _inspect_model_artifact_uncached(path: Path) -> ModelArtifactScanResult:
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
    if not manifest_path.exists():
        return ModelArtifactScanResult(
            path=path,
            kind=suffix.lstrip("."),
            status="need_confirm",
            reason="model manifest is missing",
            sha256=artifact_sha256,
            size_bytes=size_bytes,
            manifest_path=manifest_path,
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
        model_fingerprint=manifest.model_fingerprint,
        manifest=manifest,
    )


def _artifact_signature(path: Path) -> tuple[int, ...]:
    artifact = _stat_signature(path)
    manifest = _stat_signature(path.with_name("model.manifest.json"))
    return (*artifact, *manifest)


def _stat_signature(path: Path) -> tuple[int, int]:
    try:
        stat = path.stat()
    except OSError:
        return (-1, -1)
    return (int(stat.st_size), int(stat.st_mtime_ns))
