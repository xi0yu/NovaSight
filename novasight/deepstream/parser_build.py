from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import threading


logger = logging.getLogger("novasight.deepstream.parser_build")
_BUILD_LOCK = threading.Lock()
_PARSER_BUILD_ABI_VERSION = "2"


def ensure_deepstream_parser_library(
    library_path: Path | str,
    *,
    source_dir: Path | None = None,
) -> Path:
    target = Path(library_path).expanduser().resolve(strict=False)
    source = _resolve_parser_source(source_dir)
    source_fingerprint = (
        f"abi-v{_PARSER_BUILD_ABI_VERSION}:{_parser_source_fingerprint(source)}"
    )
    fingerprint_path = target.with_name(f"{target.name}.source.sha256")
    with _BUILD_LOCK:
        # Keep the lock around the complete configure/build/copy sequence so
        # concurrent runtime start requests cannot load a half-written .so.
        if target.is_file() and fingerprint_path.is_file():
            try:
                if fingerprint_path.read_text(encoding="utf-8").strip() == source_fingerprint:
                    return target
            except OSError:
                pass
        cmake = shutil.which("cmake")
        if not cmake:
            raise RuntimeError(
                "DeepStream parser auto-build requires cmake; install cmake and build-essential"
            )
        build_dir = target.parent
        build_dir.mkdir(parents=True, exist_ok=True)
        configure_command = [
            cmake,
            "-S",
            str(source),
            "-B",
            str(build_dir),
        ]
        deepstream_root = _resolve_deepstream_root()
        if deepstream_root is not None:
            configure_command.append(f"-DNOVASIGHT_DEEPSTREAM_ROOT={deepstream_root}")
        cuda_root = _resolve_cuda_root()
        if cuda_root is not None:
            configure_command.append(f"-DNOVASIGHT_CUDA_ROOT={cuda_root}")
        logger.warning(
            "DeepStream parser library missing or stale; starting automatic build target=%s",
            target,
        )
        _run_build_command(configure_command, "configure")
        _run_build_command(
            [
                cmake,
                "--build",
                str(build_dir),
                "--clean-first",
                "--parallel",
                "2",
            ],
            "compile",
        )
        built_library = build_dir / "libnovasight_parser.so"
        if not built_library.is_file():
            raise RuntimeError(
                "DeepStream parser auto-build completed without producing "
                f"{built_library}"
            )
        if built_library != target:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(built_library, target)
        fingerprint_path.write_text(f"{source_fingerprint}\n", encoding="utf-8")
        logger.info("DeepStream parser automatic build completed path=%s", target)
    return target


def _parser_source_fingerprint(source: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in source.rglob("*")
        if path.is_file()
        and path.suffix in {".cpp", ".cc", ".h", ".hpp", ".cmake", ".txt"}
    )
    for path in files:
        digest.update(path.relative_to(source).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _resolve_parser_source(source_dir: Path | None) -> Path:
    candidates: list[Path] = []
    if source_dir is not None:
        candidates.append(Path(source_dir))
    source_root = os.environ.get("NOVASIGHT_SOURCE_ROOT", "").strip()
    if source_root:
        candidates.append(Path(source_root) / "native" / "deepstream-parser")
    candidates.extend(
        [
            Path(__file__).resolve().parents[2] / "native" / "deepstream-parser",
            Path.cwd() / "native" / "deepstream-parser",
        ]
    )
    for candidate in candidates:
        resolved = candidate.expanduser().resolve(strict=False)
        if (resolved / "CMakeLists.txt").is_file():
            return resolved
    checked = ", ".join(str(item.expanduser().resolve(strict=False)) for item in candidates)
    raise RuntimeError(f"DeepStream parser source directory is unavailable; checked: {checked}")


def _resolve_deepstream_root() -> Path | None:
    candidates: list[Path] = []
    configured = os.environ.get("NOVASIGHT_DEEPSTREAM_ROOT", "").strip()
    if configured:
        candidates.append(Path(configured))
    install_root = Path("/opt/nvidia/deepstream")
    candidates.append(install_root / "deepstream")
    if install_root.is_dir():
        candidates.extend(sorted(install_root.glob("deepstream-*"), reverse=True))
    for candidate in candidates:
        resolved = candidate.expanduser().resolve(strict=False)
        if (resolved / "sources" / "includes" / "nvdsinfer_custom_impl.h").is_file():
            return resolved
    return None


def _resolve_cuda_root() -> Path | None:
    candidates: list[Path] = []
    for variable in ("NOVASIGHT_CUDA_ROOT", "CUDA_HOME", "CUDA_PATH"):
        configured = os.environ.get(variable, "").strip()
        if configured:
            candidates.append(Path(configured))
    candidates.append(Path("/usr/local/cuda"))
    usr_local = Path("/usr/local")
    if usr_local.is_dir():
        candidates.extend(sorted(usr_local.glob("cuda-*"), reverse=True))
    for candidate in candidates:
        resolved = candidate.expanduser().resolve(strict=False)
        headers = (
            resolved / "include" / "cuda_runtime_api.h",
            resolved / "targets" / "aarch64-linux" / "include" / "cuda_runtime_api.h",
        )
        if any(header.is_file() for header in headers):
            return resolved
    return None


def _run_build_command(command: list[str], stage: str) -> None:
    command_text = shlex.join(command)
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=180.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            f"DeepStream parser auto-build {stage} failed command={command_text}: {exc}"
        ) from exc
    if result.returncode == 0:
        return
    output = "\n".join(
        item.strip() for item in (result.stdout, result.stderr) if item and item.strip()
    )
    if len(output) > 6000:
        output = output[-6000:]
    raise RuntimeError(
        f"DeepStream parser auto-build {stage} failed rc={result.returncode} "
        f"command={command_text}: "
        f"{output or 'no compiler output'}"
    )


__all__ = ["ensure_deepstream_parser_library"]
