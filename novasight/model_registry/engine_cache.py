from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
import shutil
import subprocess

from .fingerprint import sha256_file, sha256_text, stable_json
from .manifest import ModelManifest


ENGINE_CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EngineBuildInfo:
    schema_version: int
    cache_key: str
    source_sha256: str
    platform: dict[str, str]
    precision: str
    input_shape: list[int]
    engine_sha256: str


class EngineCache:
    @staticmethod
    def get_engine_path(
        manifest: ModelManifest,
        *,
        cache_dir: Path = Path("models/generated"),
    ) -> Path | None:
        engine_path = EngineCache.engine_path_for(manifest, cache_dir=cache_dir)
        if not engine_path.is_file():
            return None
        if not EngineCache.valid_engine_for_current_platform(manifest, engine_path):
            return None
        return engine_path

    @staticmethod
    def engine_path_for(
        manifest: ModelManifest,
        *,
        cache_dir: Path = Path("models/generated"),
    ) -> Path:
        return Path(cache_dir) / EngineCache.compute_cache_key(manifest) / "engine.plan"

    @staticmethod
    def compute_cache_key(manifest: ModelManifest) -> str:
        payload = {
            "schema": ENGINE_CACHE_SCHEMA_VERSION,
            "model_fingerprint": manifest.model_fingerprint,
            "artifact_sha256": manifest.artifact.sha256,
            "runtime": {
                "backend": manifest.runtime.backend,
                "precision": manifest.runtime.precision,
                "batch_size": manifest.runtime.batch_size,
            },
            "input": {
                "shape": list(manifest.input.shape),
                "dtype": manifest.input.dtype,
                "layout": manifest.input.layout,
            },
            "platform": current_platform_signature(),
        }
        return sha256_text(stable_json(payload))[:32]

    @staticmethod
    def valid_engine_for_current_platform(
        manifest: ModelManifest,
        engine_path: Path,
    ) -> bool:
        engine_path = Path(engine_path)
        info_path = engine_path.with_name("build_info.json")
        if not engine_path.is_file() or not info_path.is_file():
            return False
        try:
            raw = json.loads(info_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return (
            int(raw.get("schema_version", 0)) == ENGINE_CACHE_SCHEMA_VERSION
            and raw.get("cache_key") == EngineCache.compute_cache_key(manifest)
            and raw.get("platform") == current_platform_signature()
            and raw.get("precision") == manifest.runtime.precision
            and raw.get("input_shape") == list(manifest.input.shape)
            and raw.get("engine_sha256") == sha256_file(engine_path)
        )

    @staticmethod
    def build_engine(
        manifest: ModelManifest,
        *,
        source_model_path: Path | None = None,
        cache_dir: Path = Path("models/generated"),
        trtexec: str = "trtexec",
    ) -> Path:
        source_path = Path(source_model_path or manifest.artifact.engine_path)
        engine_path = EngineCache.engine_path_for(manifest, cache_dir=cache_dir)
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.suffix.lower() in {".engine", ".plan"}:
            shutil.copyfile(source_path, engine_path)
        elif source_path.suffix.lower() == ".onnx":
            command = [
                trtexec,
                f"--onnx={source_path}",
                f"--saveEngine={engine_path}",
                f"--shapes={manifest.input.name}:{'x'.join(str(dim) for dim in manifest.input.shape)}",
            ]
            if manifest.runtime.precision.lower() == "fp16":
                command.append("--fp16")
            if manifest.runtime.precision.lower() == "int8":
                command.append("--int8")
            subprocess.run(command, check=True)
        else:
            raise ValueError(f"unsupported engine source artifact: {source_path}")
        EngineCache.write_build_info(manifest, engine_path, source_path=source_path)
        return engine_path

    @staticmethod
    def write_build_info(
        manifest: ModelManifest,
        engine_path: Path,
        *,
        source_path: Path,
    ) -> Path:
        engine_path = Path(engine_path)
        info = EngineBuildInfo(
            schema_version=ENGINE_CACHE_SCHEMA_VERSION,
            cache_key=EngineCache.compute_cache_key(manifest),
            source_sha256=sha256_file(Path(source_path)),
            platform=current_platform_signature(),
            precision=manifest.runtime.precision,
            input_shape=list(manifest.input.shape),
            engine_sha256=sha256_file(engine_path),
        )
        info_path = engine_path.with_name("build_info.json")
        info_path.write_text(
            json.dumps(asdict(info), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return info_path


def current_platform_signature() -> dict[str, str]:
    return {
        "machine": platform.machine(),
        "system": platform.system(),
        "tensorrt": _module_version("tensorrt"),
        "cuda": _module_version("cuda"),
    }


def _module_version(name: str) -> str:
    try:
        module = __import__(name)
    except Exception:
        return ""
    return str(getattr(module, "__version__", ""))


__all__ = [
    "ENGINE_CACHE_SCHEMA_VERSION",
    "EngineBuildInfo",
    "EngineCache",
    "current_platform_signature",
]
