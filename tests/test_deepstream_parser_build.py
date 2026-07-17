from pathlib import Path
import shutil
import subprocess

from novasight.deepstream import parser_build


def test_native_parser_decodes_four_output_efficient_nms(tmp_path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        return
    root = Path(__file__).resolve().parents[1]
    executable = tmp_path / "parser-contract-test"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-I",
            str(root / "native/deepstream-parser/tests"),
            str(root / "native/deepstream-parser/src/novasight_parser.cpp"),
            str(root / "native/deepstream-parser/tests/parser_contract_test.cpp"),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(executable)], check=True)


def test_missing_deepstream_parser_is_built_once(tmp_path, monkeypatch) -> None:
    source_dir = tmp_path / "native" / "deepstream-parser"
    source_dir.mkdir(parents=True)
    (source_dir / "CMakeLists.txt").write_text("project(test)", encoding="utf-8")
    library_path = tmp_path / "build" / "deepstream-parser" / "libnovasight_parser.so"
    cuda_root = tmp_path / "cuda"
    (cuda_root / "include").mkdir(parents=True)
    (cuda_root / "include" / "cuda_runtime_api.h").write_text("", encoding="utf-8")
    commands: list[list[str]] = []

    monkeypatch.setattr(parser_build.shutil, "which", lambda _name: "/usr/bin/cmake")
    monkeypatch.setenv("NOVASIGHT_CUDA_ROOT", str(cuda_root))

    def fake_run(command, **_kwargs):
        commands.append([str(item) for item in command])
        if "--build" in command:
            library_path.parent.mkdir(parents=True, exist_ok=True)
            library_path.write_bytes(b"parser")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(parser_build.subprocess, "run", fake_run)

    first = parser_build.ensure_deepstream_parser_library(
        library_path,
        source_dir=source_dir,
    )
    second = parser_build.ensure_deepstream_parser_library(
        library_path,
        source_dir=source_dir,
    )

    assert first == library_path.resolve()
    assert second == first
    assert len(commands) == 2
    assert commands[0][:2] == ["/usr/bin/cmake", "-S"]
    assert f"-DNOVASIGHT_CUDA_ROOT={cuda_root.resolve()}" in commands[0]
    assert commands[1][:2] == ["/usr/bin/cmake", "--build"]


def test_stale_parser_forces_clean_rebuild_before_recording_new_fingerprint(
    tmp_path,
    monkeypatch,
) -> None:
    source_dir = tmp_path / "native" / "deepstream-parser"
    source_dir.mkdir(parents=True)
    (source_dir / "CMakeLists.txt").write_text("project(test)", encoding="utf-8")
    (source_dir / "parser.cpp").write_text("new parser ABI", encoding="utf-8")
    library_path = tmp_path / "build" / "deepstream-parser" / "libnovasight_parser.so"
    library_path.parent.mkdir(parents=True)
    library_path.write_bytes(b"old parser without new symbols")
    library_path.with_name(f"{library_path.name}.source.sha256").write_text(
        parser_build._parser_source_fingerprint(source_dir),
        encoding="utf-8",
    )
    commands: list[list[str]] = []

    monkeypatch.setattr(parser_build.shutil, "which", lambda _name: "/usr/bin/cmake")

    def fake_run(command, **_kwargs):
        normalized = [str(item) for item in command]
        commands.append(normalized)
        # This models the Jetson failure: an incremental build trusts preserved
        # mtimes and leaves the old .so untouched. A clean build produces the ABI.
        if "--build" in normalized and "--clean-first" in normalized:
            library_path.write_bytes(b"new parser with explicit entrypoints")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(parser_build.subprocess, "run", fake_run)

    parser_build.ensure_deepstream_parser_library(
        library_path,
        source_dir=source_dir,
    )

    assert library_path.read_bytes() == b"new parser with explicit entrypoints"
    assert "--clean-first" in commands[1]


def test_deepstream_parser_build_failure_includes_compiler_output(tmp_path, monkeypatch) -> None:
    source_dir = tmp_path / "native" / "deepstream-parser"
    source_dir.mkdir(parents=True)
    (source_dir / "CMakeLists.txt").write_text("project(test)", encoding="utf-8")
    library_path = tmp_path / "build" / "deepstream-parser" / "libnovasight_parser.so"

    monkeypatch.setattr(parser_build.shutil, "which", lambda _name: "/usr/bin/cmake")
    monkeypatch.setattr(
        parser_build.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command,
            1,
            stdout="configure output",
            stderr="missing nvdsinfer_custom_impl.h",
        ),
    )

    try:
        parser_build.ensure_deepstream_parser_library(
            library_path,
            source_dir=source_dir,
        )
    except RuntimeError as exc:
        assert "missing nvdsinfer_custom_impl.h" in str(exc)
    else:
        raise AssertionError("failed parser build must be reported")
