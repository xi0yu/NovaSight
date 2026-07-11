from pathlib import Path
import subprocess

from novasight.deepstream import parser_build


def test_missing_deepstream_parser_is_built_once(tmp_path, monkeypatch) -> None:
    source_dir = tmp_path / "native" / "deepstream-parser"
    source_dir.mkdir(parents=True)
    (source_dir / "CMakeLists.txt").write_text("project(test)", encoding="utf-8")
    library_path = tmp_path / "build" / "deepstream-parser" / "libnovasight_parser.so"
    commands: list[list[str]] = []

    monkeypatch.setattr(parser_build.shutil, "which", lambda _name: "/usr/bin/cmake")

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
    assert commands[1][:2] == ["/usr/bin/cmake", "--build"]


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
