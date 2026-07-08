from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import uvicorn

from .api import create_app
from .config import load_runtime_config
from .runtime import configure_logging


_CAPTURE_PROFILE_RE = re.compile(
    r"^\s*(?P<pixel_format>\S+)\s+"
    r"(?P<width>\d+)x(?P<height>\d+)@(?P<fps>\d+)(?:\b|$)"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("novasight")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--config", default="config/novasight.yaml")
    parser.add_argument("--data-dir", default="data")
    subparsers = parser.add_subparsers(dest="command")
    doctor = subparsers.add_parser("doctor")
    doctor_sub = doctor.add_subparsers(dest="doctor_command")
    doctor_camera = doctor_sub.add_parser("camera")
    doctor_camera.add_argument("--device", default="/dev/video0")
    doctor_jetson = doctor_sub.add_parser("jetson-bridge")
    doctor_jetson.add_argument("--module", default=None)
    doctor_jetson_build = doctor_sub.add_parser("jetson-native-build")
    doctor_jetson_build.add_argument("--build-dir", default="build/jetson-native")
    doctor_jetson_build.add_argument("--source", default=None)
    doctor_jetson_build.add_argument("--cmake", default="cmake")
    doctor_jetson_build.add_argument("--skip-load", action="store_true")
    doctor_jetson_smoke = doctor_sub.add_parser("jetson-native-smoke")
    doctor_jetson_smoke.add_argument("--device", default=None)
    doctor_jetson_smoke.add_argument("--preference", default=None)
    doctor_jetson_smoke.add_argument("--pixel-format", default=None)
    doctor_jetson_smoke.add_argument("--width", type=int, default=None)
    doctor_jetson_smoke.add_argument("--height", type=int, default=None)
    doctor_jetson_smoke.add_argument("--fps", type=int, default=None)
    doctor_jetson_smoke.add_argument("--roi-size", type=int, default=None)
    doctor_jetson_smoke.add_argument("--roi-offset-x", type=int, default=None)
    doctor_jetson_smoke.add_argument("--roi-offset-y", type=int, default=None)
    doctor_jetson_smoke.add_argument("--input-shape", default=None)
    doctor_jetson_smoke.add_argument("--dtype", default=None)
    doctor_jetson_smoke.add_argument("--timeout", type=float, default=5.0)
    doctor_jetson_smoke.add_argument("--library", default=None)
    doctor_jetson_smoke.add_argument("--module", default=None)
    doctor_jetson_smoke.add_argument("--tensorrt-engine", default=None)
    doctor_jetson_smoke.add_argument("--classes", default=None)
    doctor_jetson_zero_copy = doctor_sub.add_parser("jetson-zero-copy")
    doctor_jetson_zero_copy.add_argument("--build-dir", default="build/jetson-native")
    doctor_jetson_zero_copy.add_argument("--source", default=None)
    doctor_jetson_zero_copy.add_argument("--cmake", default="cmake")
    doctor_jetson_zero_copy.add_argument("--device", default=None)
    doctor_jetson_zero_copy.add_argument("--preference", default=None)
    doctor_jetson_zero_copy.add_argument("--pixel-format", default=None)
    doctor_jetson_zero_copy.add_argument("--width", type=int, default=None)
    doctor_jetson_zero_copy.add_argument("--height", type=int, default=None)
    doctor_jetson_zero_copy.add_argument("--fps", type=int, default=None)
    doctor_jetson_zero_copy.add_argument("--roi-size", type=int, default=None)
    doctor_jetson_zero_copy.add_argument("--roi-offset-x", type=int, default=None)
    doctor_jetson_zero_copy.add_argument("--roi-offset-y", type=int, default=None)
    doctor_jetson_zero_copy.add_argument("--input-shape", default=None)
    doctor_jetson_zero_copy.add_argument("--dtype", default=None)
    doctor_jetson_zero_copy.add_argument("--timeout", type=float, default=5.0)
    doctor_jetson_zero_copy.add_argument("--module", default=None)
    doctor_jetson_zero_copy.add_argument("--tensorrt-engine", default=None)
    doctor_jetson_zero_copy.add_argument("--classes", default=None)
    doctor_jetson_zero_copy.add_argument(
        "--require-tensorrt-engine",
        action="store_true",
        help="Fail one-shot acceptance unless --tensorrt-engine is provided.",
    )
    doctor_jetson_zero_copy.add_argument("--report-json", default=None)
    doctor_jetson_zero_copy_report = doctor_sub.add_parser("jetson-zero-copy-report")
    doctor_jetson_zero_copy_report.add_argument("--report-json", required=True)
    doctor_jetson_zero_copy_report.add_argument(
        "--require-tensorrt-engine",
        action="store_true",
        help="Reject reports that do not include TensorRT engine binding evidence.",
    )
    doctor_deepstream_smoke = doctor_sub.add_parser("deepstream-smoke")
    doctor_deepstream_smoke.add_argument("--manifest", required=True)
    doctor_deepstream_smoke.add_argument("--nvinfer-config", required=True)
    doctor_deepstream_smoke.add_argument("--device", default="/dev/video0")
    doctor_deepstream_smoke.add_argument("--capture-width", type=int, default=1920)
    doctor_deepstream_smoke.add_argument("--capture-height", type=int, default=1080)
    doctor_deepstream_smoke.add_argument("--fps", type=int, default=120)
    doctor_deepstream_smoke.add_argument("--roi-left", type=int, default=720)
    doctor_deepstream_smoke.add_argument("--roi-top", type=int, default=300)
    doctor_deepstream_smoke.add_argument("--roi-size", type=int, default=480)
    doctor_deepstream_smoke.add_argument("--io-mode", type=int, default=2)
    doctor_deepstream_smoke.add_argument("--batched-push-timeout-us", type=int, default=0)
    doctor_deepstream_smoke.add_argument("--tracker-config", default="")
    doctor_deepstream_smoke.add_argument("--seconds", type=float, default=5.0)
    doctor_deepstream_smoke.add_argument("--poll-interval", type=float, default=0.02)
    doctor_deepstream_smoke.add_argument("--report-json", default=None)
    doctor_deepstream_smoke_report = doctor_sub.add_parser("deepstream-smoke-report")
    doctor_deepstream_smoke_report.add_argument("--report-json", required=True)
    doctor_deepstream_smoke_report.add_argument("--min-tensor-meta-fps", type=float, default=0.0)
    doctor_deepstream_smoke_report.add_argument("--min-postprocess-fps", type=float, default=0.0)
    doctor_deepstream_smoke_report.add_argument("--min-detection-batch-fps", type=float, default=0.0)
    doctor_deepstream_smoke_report.add_argument("--max-frame-age-ms", type=float, default=0.0)
    doctor_kmnet = doctor_sub.add_parser("kmnet")
    doctor_kmnet.add_argument("--km-host", default=None)
    doctor_kmnet.add_argument("--km-port", type=int, default=None)
    doctor_kmnet.add_argument("--km-uuid", default=None)
    doctor_kmnet.add_argument("--monitor-port", type=int, default=None)
    doctor_kmnet.add_argument("--move-dx", type=int, default=1)
    doctor_kmnet.add_argument("--move-dy", type=int, default=0)
    doctor_kmnet.add_argument("--no-move", action="store_true")

    capture_smoke = subparsers.add_parser("capture-smoke")
    capture_smoke.add_argument("--device", default="/dev/video0")
    capture_smoke.add_argument("--seconds", type=float, default=5.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_runtime_config(args.config)
    if args.host is not None:
        cfg.web.host = args.host
    if args.port is not None:
        cfg.web.port = args.port
    configure_logging(cfg)
    if args.command == "doctor" and args.doctor_command is None:
        print(
            "error: doctor requires a subcommand: camera, jetson-bridge, "
            "jetson-native-build, jetson-native-smoke, jetson-zero-copy, "
            "jetson-zero-copy-report, deepstream-smoke, deepstream-smoke-report, kmnet",
            file=sys.stderr,
        )
        return 2
    if args.command == "doctor" and args.doctor_command == "camera":
        from novasight.capture import query_capabilities
        from novasight.capture.profile import select_capture_profile

        caps = query_capabilities(args.device)
        print(f"device: {caps.device}")
        print(f"available: {caps.available}")
        for cap in caps.capabilities:
            print(f"{cap.pixel_format} {cap.width}x{cap.height} fps={cap.fps_list}")
        if caps.available:
            profile = select_capture_profile(args.device, caps.capabilities)
            print(
                f"selected: {profile.pixel_format} "
                f"{profile.width}x{profile.height}@{profile.fps}"
            )
        elif caps.reason:
            print(f"reason: {caps.reason}")
        return 0 if caps.available else 2
    if args.command == "doctor" and args.doctor_command == "jetson-bridge":
        from novasight.inference.jetson import JetsonGpuResourcePreprocessor

        kwargs = {"module_name": args.module} if args.module else {}
        preprocessor = JetsonGpuResourcePreprocessor(**kwargs)
        status = preprocessor.status()
        print(f"selected: {status['selected']}")
        print(f"enabled: {status['enabled']}")
        print(f"available: {status['available']}")
        print(f"module: {status['module']}")
        print(f"env: {status['env']}")
        print(f"required_abi_version: {status['required_abi_version']}")
        print(f"abi_version: {status['abi_version']}")
        print(f"abi_compatible: {status['abi_compatible']}")
        print(f"native_ready: {status['native_ready']}")
        print(f"native_status: {status['native_status']}")
        print(f"capabilities: {status['capabilities']}")
        print(f"contract: {status['contract']}")
        if status["reason"]:
            print(f"reason: {status['reason']}")
        if status["detail"]:
            print(f"detail: {status['detail']}")
        return 0 if status["available"] else 2
    if args.command == "doctor" and args.doctor_command == "jetson-native-build":
        return _doctor_jetson_native_build(args)
    if args.command == "doctor" and args.doctor_command == "jetson-native-smoke":
        return _doctor_jetson_native_smoke(args, cfg)
    if args.command == "doctor" and args.doctor_command == "jetson-zero-copy":
        return _doctor_jetson_zero_copy(args, cfg)
    if args.command == "doctor" and args.doctor_command == "jetson-zero-copy-report":
        return _doctor_jetson_zero_copy_report(args)
    if args.command == "doctor" and args.doctor_command == "deepstream-smoke":
        return _doctor_deepstream_smoke(args)
    if args.command == "doctor" and args.doctor_command == "deepstream-smoke-report":
        return _doctor_deepstream_smoke_report(args)
    if args.command == "doctor" and args.doctor_command == "kmnet":
        from novasight.executors.kmnet import KmNetExecutor

        host = args.km_host or cfg.hardware.host
        port = args.km_port if args.km_port is not None else cfg.hardware.port
        uuid = args.km_uuid if args.km_uuid is not None else cfg.hardware.uuid
        monitor_port = (
            args.monitor_port
            if args.monitor_port is not None
            else cfg.hardware.monitor_port
        )
        executor = KmNetExecutor(
            host=host,
            port=port,
            uuid=uuid,
            monitor_port=monitor_port,
        )
        status = executor.status()
        print(f"driver_available: {status['available']}")
        print(f"connected: {status['connected']}")
        print(f"host: {status['host']}")
        print(f"port: {status['port']}")
        print(f"monitor_port: {status['monitor_port']}")
        print(f"monitoring: {status['monitoring']}")
        print(f"has_move_auto: {status['has_move_auto']}")
        print(f"has_move_bezier: {status['has_move_bezier']}")
        print(f"has_trace: {status['has_trace']}")
        print(f"has_left_button: {status['has_left_button']}")
        print(f"has_right_button: {status['has_right_button']}")
        buttons = executor.read_buttons()
        print(f"buttons_available: {buttons['available']}")
        print(f"button_left: {buttons['left']}")
        print(f"button_right: {buttons['right']}")
        if buttons["reason"]:
            print(f"button_reason: {buttons['reason']}")
        if not args.no_move:
            result = executor.diagnostic_move(args.move_dx, args.move_dy)
            print(f"move_sent: {result.sent}")
            print(f"move_message: {result.message}")
        refreshed = executor.status()
        print(f"move_count: {refreshed['move_count']}")
        print(f"last_dx: {refreshed['last_dx']}")
        print(f"last_dy: {refreshed['last_dy']}")
        if refreshed["last_error"]:
            print(f"reason: {refreshed['last_error']}")
        return 0 if refreshed["available"] and refreshed["connected"] else 2
    if args.command == "capture-smoke":
        from novasight.capture.service import CaptureService

        cfg.capture.device = args.device
        service = CaptureService(config=cfg.capture)
        try:
            state = service.configure(args.device)
            print(f"device: {state.device}")
            print(f"available: {state.available}")
            if state.profile is not None:
                profile = state.profile
                print(
                    f"selected: {profile.pixel_format} "
                    f"{profile.width}x{profile.height}@{profile.fps}"
                )
            if state.backend:
                print(f"backend: {state.backend}")
            if not state.available:
                if state.last_error:
                    print(f"reason: {state.last_error}")
                return 2
            start = time.monotonic()
            last_frame_id = 0
            while time.monotonic() - start < args.seconds:
                frame = service.wait_preview_frame(
                    after_frame_id=last_frame_id,
                    timeout_s=max(service.empty_read_sleep_s, 0.001),
                )
                if frame is not None:
                    last_frame_id = frame.frame_id
            state = service.state
            print(f"fps_capture: {state.fps_capture:.2f}")
            print(f"capture_wait_ms: {state.capture_wait_ms:.2f}")
            print(f"frame_period_ms: {state.frame_period_ms:.2f}")
            print(f"frames_dropped: {state.frames_dropped}")
            return 0
        except Exception as exc:
            print(f"reason: {exc}")
            return 2
        finally:
            service.stop("capture smoke complete")
    app = create_app(
        data_dir=Path(args.data_dir),
        config_path=Path(args.config),
        config=cfg,
    )
    uvicorn.run(app, host=cfg.web.host, port=cfg.web.port)
    return 0


def _doctor_jetson_native_build(args: argparse.Namespace) -> int:
    import novasight_jetson_preprocess_native as native_backend

    native_dir = Path(native_backend.__file__).resolve().parent / "native"
    if not native_dir.is_dir():
        print(f"native_dir: {native_dir}")
        print("available: False")
        print("reason: jetson_native_source_tree_missing")
        return 2
    production_source = (
        Path(args.source).expanduser()
        if args.source
        else native_dir / "src/jetson/novasight_jetson_preprocess_native_jetson_cuda.cu"
    )
    build_dir = Path(args.build_dir).expanduser()
    if not production_source.is_file():
        print(f"native_dir: {native_dir}")
        print(f"production_source: {production_source}")
        print("available: False")
        print("reason: jetson_native_source_missing")
        return 2
    configure_command = [
        args.cmake,
        "-S",
        str(native_dir),
        "-B",
        str(build_dir),
        "-DNOVASIGHT_JETSON_PREPROCESS_IMPL=jetson",
        f"-DNOVASIGHT_JETSON_PREPROCESS_PRODUCTION_SOURCE={production_source}",
    ]
    build_command = [args.cmake, "--build", str(build_dir)]
    print(f"native_dir: {native_dir}")
    print(f"production_source: {production_source}")
    print(f"build_dir: {build_dir}")
    print(f"configure_command: {' '.join(configure_command)}")
    configure_result = _run_doctor_command(configure_command)
    print(f"configure_exit: {configure_result.returncode}")
    if configure_result.returncode != 0:
        print("available: False")
        print("reason: jetson_native_configure_failed")
        return 2
    print(f"build_command: {' '.join(build_command)}")
    build_result = _run_doctor_command(build_command)
    print(f"build_exit: {build_result.returncode}")
    if build_result.returncode != 0:
        print("available: False")
        print("reason: jetson_native_build_failed")
        return 2

    library = build_dir / "libnovasight_jetson_preprocess_native.so"
    print(f"library: {library}")
    if not library.is_file():
        print("available: False")
        print("reason: jetson_native_library_missing")
        return 2
    if args.skip_load:
        print("load_check: skipped")
        return 0

    previous_library = os.environ.get(native_backend.LIBRARY_ENV)
    os.environ[native_backend.LIBRARY_ENV] = str(library)
    try:
        reset = getattr(native_backend, "_reset_library_cache", None)
        if callable(reset):
            reset()
        status = native_backend.status()
    finally:
        if previous_library is None:
            os.environ.pop(native_backend.LIBRARY_ENV, None)
        else:
            os.environ[native_backend.LIBRARY_ENV] = previous_library
        reset = getattr(native_backend, "_reset_library_cache", None)
        if callable(reset):
            reset()
    print(f"status_available: {status.get('available')}")
    print(f"status_ready: {status.get('ready')}")
    print(f"status_backend: {status.get('backend')}")
    print(f"status_zero_copy: {status.get('zero_copy')}")
    print(f"status_memory_space: {status.get('memory_space')}")
    if status.get("reason"):
        print(f"reason: {status.get('reason')}")
    if status.get("detail"):
        print(f"detail: {status.get('detail')}")
    if not _jetson_native_status_is_accepted(status):
        print("available: False")
        print("reason: jetson_native_status_contract_failed")
        return 2
    return 0


def _doctor_jetson_zero_copy(args: argparse.Namespace, cfg: object) -> int:
    build_dir = Path(args.build_dir).expanduser()
    library = build_dir / "libnovasight_jetson_preprocess_native.so"
    report = _jetson_zero_copy_report(args, cfg=cfg, library=library)
    if bool(getattr(args, "require_tensorrt_engine", False)) and not str(
        args.tensorrt_engine or ""
    ).strip():
        print("available: False")
        print("reason: jetson_zero_copy_tensorrt_engine_required")
        report["accepted"] = False
        report["reason"] = "jetson_zero_copy_tensorrt_engine_required"
        _finish_and_write_report(args.report_json, report)
        return 2
    print("phase: jetson-native-build")
    build_result, build_stdout = _run_doctor_phase(
        _doctor_jetson_native_build,
        argparse.Namespace(
            build_dir=args.build_dir,
            source=args.source,
            cmake=args.cmake,
            skip_load=False,
        ),
    )
    report["phases"]["build"]["exit_code"] = build_result
    report["phases"]["build"]["accepted"] = build_result == 0
    report["phases"]["build"]["stdout"] = build_stdout
    if build_result != 0:
        print("available: False")
        print("reason: jetson_zero_copy_build_failed")
        report["accepted"] = False
        report["reason"] = "jetson_zero_copy_build_failed"
        _finish_and_write_report(args.report_json, report)
        return build_result
    print("phase: jetson-native-smoke")
    smoke_result, smoke_stdout = _run_doctor_phase(
        _doctor_jetson_native_smoke,
        argparse.Namespace(
            device=args.device,
            preference=args.preference,
            pixel_format=args.pixel_format,
            width=args.width,
            height=args.height,
            fps=args.fps,
            roi_size=args.roi_size,
            roi_offset_x=args.roi_offset_x,
            roi_offset_y=args.roi_offset_y,
            input_shape=args.input_shape,
            dtype=args.dtype,
            timeout=args.timeout,
            library=str(library),
            module=args.module,
            tensorrt_engine=args.tensorrt_engine,
            classes=args.classes,
        ),
        cfg,
    )
    report["phases"]["smoke"]["exit_code"] = smoke_result
    report["phases"]["smoke"]["accepted"] = smoke_result == 0
    report["phases"]["smoke"]["stdout"] = smoke_stdout
    if smoke_result != 0:
        print("available: False")
        print("reason: jetson_zero_copy_smoke_failed")
        report["accepted"] = False
        report["reason"] = "jetson_zero_copy_smoke_failed"
        _finish_and_write_report(args.report_json, report)
        return smoke_result
    report["accepted"] = True
    report["reason"] = ""
    report["finished_at"] = datetime.now(UTC).isoformat()
    failures = _validate_jetson_zero_copy_report(
        report,
        require_tensorrt_engine=bool(getattr(args, "require_tensorrt_engine", False)),
    )
    if failures:
        print("available: False")
        print("reason: jetson_zero_copy_report_invalid")
        for failure in failures:
            print(f"missing: {failure}")
        report["accepted"] = False
        report["reason"] = "jetson_zero_copy_report_invalid"
        report["validation_failures"] = failures
        _finish_and_write_report(args.report_json, report)
        return 2
    report["validation_failures"] = []
    print("zero_copy_acceptance: True")
    _finish_and_write_report(args.report_json, report)
    return 0


def _jetson_zero_copy_report(
    args: argparse.Namespace,
    *,
    cfg: object,
    library: Path,
) -> dict[str, object]:
    roi_size = int(args.roi_size or cfg.roi.size)
    roi_offset_x = int(
        args.roi_offset_x if args.roi_offset_x is not None else cfg.roi.offset_x
    )
    roi_offset_y = int(
        args.roi_offset_y if args.roi_offset_y is not None else cfg.roi.offset_y
    )
    input_shape = _doctor_tensor_input_shape(
        args.input_shape,
        dtype=args.dtype,
        roi_size=roi_size,
    )
    return {
        "schema_version": 1,
        "check": "jetson-zero-copy",
        "started_at": datetime.now(UTC).isoformat(),
        "accepted": False,
        "reason": "not_finished",
        "validation_failures": [],
        "parameters": {
            "build_dir": str(Path(args.build_dir).expanduser()),
            "source": str(Path(args.source).expanduser()) if args.source else "bundled",
            "cmake": args.cmake,
            "library": str(library),
            "device": args.device,
            "preference": args.preference,
            "pixel_format": args.pixel_format,
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "roi_size": roi_size,
            "roi_offset_x": roi_offset_x,
            "roi_offset_y": roi_offset_y,
            "input_shape": str(input_shape),
            "dtype": input_shape.dtype,
            "timeout": args.timeout,
            "module": args.module,
            "tensorrt_engine": args.tensorrt_engine,
            "classes": args.classes,
        },
        "phases": {
            "build": {
                "name": "jetson-native-build",
                "exit_code": None,
                "accepted": False,
                "stdout": "",
            },
            "smoke": {
                "name": "jetson-native-smoke",
                "exit_code": None,
                "accepted": False,
                "stdout": "",
            },
        },
    }


def _run_doctor_phase(function, *args) -> tuple[int, str]:
    tee = _StdoutCaptureTee(sys.stdout)
    with contextlib.redirect_stdout(tee):
        result = function(*args)
    return int(result), tee.getvalue()


class _StdoutCaptureTee:
    def __init__(self, stream: object) -> None:
        self._stream = stream
        self._buffer = io.StringIO()

    def write(self, text: str) -> int:
        self._buffer.write(text)
        return self._stream.write(text)  # type: ignore[attr-defined]

    def flush(self) -> None:
        flush = getattr(self._stream, "flush", None)
        if callable(flush):
            flush()

    def getvalue(self) -> str:
        return self._buffer.getvalue()


def _finish_and_write_report(path: str | None, report: dict[str, object]) -> None:
    report["finished_at"] = datetime.now(UTC).isoformat()
    if not path:
        return
    report_path = Path(path).expanduser()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"report_json: {report_path}")


def _doctor_jetson_zero_copy_report(args: argparse.Namespace) -> int:
    report_path = Path(args.report_json).expanduser()
    print(f"report_json: {report_path}")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print("accepted: False")
        print("reason: jetson_zero_copy_report_unreadable")
        print(f"detail: {exc}")
        return 2
    failures = _validate_jetson_zero_copy_report(
        report,
        require_tensorrt_engine=bool(args.require_tensorrt_engine),
    )
    if failures:
        print("accepted: False")
        print("reason: jetson_zero_copy_report_invalid")
        for failure in failures:
            print(f"missing: {failure}")
        return 2
    phases = report["phases"]
    build = phases["build"]
    smoke = phases["smoke"]
    print("accepted: True")
    print(f"library: {report['parameters']['library']}")
    print(f"build_exit: {build['exit_code']}")
    print(f"smoke_exit: {smoke['exit_code']}")
    _print_zero_copy_report_evidence(report)
    return 0


def _doctor_deepstream_smoke(args: argparse.Namespace) -> int:
    from novasight.api.deepstream_runtime import (
        manifest_input_hw,
        validate_deepstream_config_matches_manifest,
    )
    from novasight.deepstream import (
        DeepStreamDetectionBackend,
        DeepStreamPipelineConfig,
        check_deepstream_dependencies,
    )
    from novasight.model_registry.deepstream_config import read_nvinfer_config_fingerprint
    from novasight.model_registry.manifest import (
        read_manifest,
        validate_manifest_engine_artifact,
    )

    manifest_path = Path(args.manifest).expanduser()
    nvinfer_config_path = Path(args.nvinfer_config).expanduser()
    report = _deepstream_smoke_report(args)
    print(f"manifest: {manifest_path}")
    print(f"nvinfer_config: {nvinfer_config_path}")
    if not manifest_path.is_file():
        print("available: False")
        print("reason: deepstream_manifest_missing")
        return _finish_deepstream_smoke_report(args, report, 2, "deepstream_manifest_missing")
    if not nvinfer_config_path.is_file():
        print("available: False")
        print("reason: deepstream_nvinfer_config_missing")
        return _finish_deepstream_smoke_report(
            args,
            report,
            2,
            "deepstream_nvinfer_config_missing",
        )
    try:
        manifest = read_manifest(manifest_path)
        engine_path = manifest_path.with_name(manifest.artifact.engine_path)
        validate_manifest_engine_artifact(manifest, engine_path)
        validate_deepstream_config_matches_manifest(
            manifest,
            nvinfer_config_path=nvinfer_config_path,
            engine_path=engine_path,
        )
        actual_fingerprint = read_nvinfer_config_fingerprint(nvinfer_config_path)
        model_height, model_width = manifest_input_hw(manifest)
        pipeline_config = DeepStreamPipelineConfig(
            device=args.device,
            capture_width=args.capture_width,
            capture_height=args.capture_height,
            fps=args.fps,
            roi_left=args.roi_left,
            roi_top=args.roi_top,
            roi_size=args.roi_size,
            model_width=model_width,
            model_height=model_height,
            nvinfer_config_path=nvinfer_config_path,
            pixel_format="MJPG",
            io_mode=args.io_mode,
            batched_push_timeout_us=args.batched_push_timeout_us,
            tracker_config_path=Path(args.tracker_config).expanduser() if args.tracker_config else None,
        )
        backend = DeepStreamDetectionBackend(
            pipeline_config=pipeline_config,
            manifest=manifest,
            roi_width=args.roi_size,
            roi_height=args.roi_size,
        )
    except Exception as exc:
        print("available: False")
        print(f"reason: {exc}")
        return _finish_deepstream_smoke_report(args, report, 2, str(exc))

    dependency = check_deepstream_dependencies()
    print(f"dependency_available: {dependency.available}")
    if dependency.reason:
        print(f"dependency_reason: {dependency.reason}")
    if dependency.detail:
        print(f"dependency_detail: {dependency.detail}")
    print(f"model_id: {manifest.model_id}")
    print(f"engine: {engine_path}")
    print(f"model_fingerprint: {manifest.model_fingerprint}")
    print(f"nvinfer_config_fingerprint: {actual_fingerprint}")
    print(f"runtime_precision: {manifest.runtime.precision}")
    print(f"model_input: {model_width}x{model_height}")
    print(f"input_color_format: {manifest.input.color_format}")
    print(f"input_scale_factor: {manifest.input.scale_factor}")
    print(f"maintain_aspect_ratio: {manifest.input.maintain_aspect_ratio}")
    print(f"symmetric_padding: {manifest.input.symmetric_padding}")
    print(f"model_output: {manifest.output.name} {manifest.output.shape}")
    print(f"output_class_count: {manifest.output.class_count}")
    print(f"output_class_names: {','.join(manifest.output.class_names) or '-'}")
    print(f"parser: {manifest.postprocess.parser}")
    print(f"confidence_threshold: {manifest.postprocess.confidence_threshold}")
    print(f"nms_threshold: {manifest.postprocess.nms_iou_threshold}")
    print(f"capture_pixel_format: {pipeline_config.pixel_format}")
    print(f"io_mode: {pipeline_config.io_mode}")
    print(f"batched_push_timeout_us: {pipeline_config.batched_push_timeout_us}")
    print(f"roi: {args.roi_left},{args.roi_top},{args.roi_size}x{args.roi_size}")
    print(f"pipeline: {backend.pipeline_description}")
    report["evidence"].update(
        {
            "dependency_available": dependency.available,
            "dependency_reason": dependency.reason,
            "dependency_detail": dependency.detail,
            "model_id": manifest.model_id,
            "engine": str(engine_path),
            "model_fingerprint": manifest.model_fingerprint,
            "nvinfer_config_fingerprint": actual_fingerprint,
            "model_runtime": {
                "backend": manifest.runtime.backend,
                "precision": manifest.runtime.precision,
                "batch_size": manifest.runtime.batch_size,
            },
            "model_input": {
                "name": manifest.input.name,
                "shape": list(manifest.input.shape),
                "dtype": manifest.input.dtype,
                "layout": manifest.input.layout,
                "color_format": manifest.input.color_format,
                "scale_factor": manifest.input.scale_factor,
                "maintain_aspect_ratio": manifest.input.maintain_aspect_ratio,
                "symmetric_padding": manifest.input.symmetric_padding,
                "width": model_width,
                "height": model_height,
            },
            "model_output": {
                "name": manifest.output.name,
                "shape": list(manifest.output.shape),
                "class_count": manifest.output.class_count,
                "class_names": list(manifest.output.class_names),
            },
            "parser": manifest.postprocess.parser,
            "confidence_threshold": manifest.postprocess.confidence_threshold,
            "nms_threshold": manifest.postprocess.nms_iou_threshold,
            "capture_pixel_format": pipeline_config.pixel_format,
            "io_mode": pipeline_config.io_mode,
            "batched_push_timeout_us": pipeline_config.batched_push_timeout_us,
            "roi": {
                "left": args.roi_left,
                "top": args.roi_top,
                "size": args.roi_size,
            },
            "pipeline": backend.pipeline_description,
        }
    )
    if not dependency.available:
        print("available: False")
        print(f"reason: {dependency.reason or 'deepstream_dependency_unavailable'}")
        return _finish_deepstream_smoke_report(
            args,
            report,
            2,
            dependency.reason or "deepstream_dependency_unavailable",
        )

    started = False
    try:
        backend.start()
        started = True
        deadline = time.monotonic() + max(0.0, float(args.seconds))
        last_frame_id = -1
        observed = 0
        while time.monotonic() < deadline:
            result = backend.latest_result(after_frame_id=last_frame_id)
            if result is not None:
                observed += 1
                last_frame_id = result.frame_id
            time.sleep(max(0.001, float(args.poll_interval)))
        status = backend.status()
        latest = backend.latest_result()
        if observed <= 0 and latest is not None:
            observed = 1
        print(f"available: {status['available']}")
        print(f"running: {status['running']}")
        print(f"published_batches: {status['published_batches']}")
        print(f"observed_batches: {observed}")
        print(f"terminal_error: {bool(status.get('terminal_error', False))}")
        print(f"tensor_meta_frames: {int(status.get('tensor_meta_frames', 0))}")
        print(f"postprocess_frames: {int(status.get('postprocess_frames', 0))}")
        print(f"window_tensor_meta_frames: {int(status.get('window_tensor_meta_frames', 0))}")
        print(f"window_postprocess_frames: {int(status.get('window_postprocess_frames', 0))}")
        print(f"tensor_meta_fps: {float(status.get('tensor_meta_fps', 0.0)):.2f}")
        print(f"postprocess_fps: {float(status.get('postprocess_fps', 0.0)):.2f}")
        print(f"detection_batch_fps: {float(status.get('detection_batch_fps', 0.0)):.2f}")
        print(f"last_frame_id: {status['last_frame_id']}")
        print(f"last_frame_age_ms: {float(status.get('last_frame_age_ms', 0.0)):.2f}")
        print(f"last_inference_latency_ms: {float(status.get('last_inference_latency_ms', 0.0)):.2f}")
        print(f"latency_source: {status.get('latency_source', '')}")
        print(f"timestamp_source: {status.get('timestamp_source', '')}")
        print(
            "capture_to_tensor_meta_ms: "
            f"{float(status.get('last_inference_latency_ms', 0.0)):.2f}"
        )
        latency_stats = status.get("capture_to_tensor_meta_ms_stats", {})
        if isinstance(latency_stats, dict):
            print(
                "capture_to_tensor_meta_ms_p50: "
                f"{float(latency_stats.get('p50', 0.0)):.2f}"
            )
            print(
                "capture_to_tensor_meta_ms_p95: "
                f"{float(latency_stats.get('p95', 0.0)):.2f}"
            )
            print(
                "capture_to_tensor_meta_ms_p99: "
                f"{float(latency_stats.get('p99', 0.0)):.2f}"
            )
        print(f"last_detection_count: {status.get('last_detection_count', 0)}")
        print(f"last_error: {status['last_error']}")
        report["evidence"].update(
            {
                "available": status["available"],
                "running": status["running"],
                "published_batches": status["published_batches"],
                "observed_batches": observed,
                "terminal_error": bool(status.get("terminal_error", False)),
                "tensor_meta_frames": int(status.get("tensor_meta_frames", 0)),
                "postprocess_frames": int(status.get("postprocess_frames", 0)),
                "window_tensor_meta_frames": int(status.get("window_tensor_meta_frames", 0)),
                "window_postprocess_frames": int(status.get("window_postprocess_frames", 0)),
                "tensor_meta_fps": float(status.get("tensor_meta_fps", 0.0)),
                "postprocess_fps": float(status.get("postprocess_fps", 0.0)),
                "detection_batch_fps": float(status.get("detection_batch_fps", 0.0)),
                "last_frame_id": status["last_frame_id"],
                "last_frame_age_ms": float(status.get("last_frame_age_ms", 0.0)),
                "last_inference_latency_ms": float(
                    status.get("last_inference_latency_ms", 0.0)
                ),
                "capture_to_tensor_meta_ms_stats": status.get(
                    "capture_to_tensor_meta_ms_stats",
                    {},
                ),
                "runtime_model_runtime": status.get("model_runtime", {}),
                "runtime_model_input": status.get("model_input", {}),
                "runtime_model_output": status.get("model_output", {}),
                "latency_source": status.get("latency_source", ""),
                "timestamp_source": status.get("timestamp_source", ""),
                "last_detection_count": status.get("last_detection_count", 0),
                "last_error": status["last_error"],
            }
        )
        if latest is not None:
            print(f"last_capture_ts_ns: {latest.capture_ts_ns}")
            print(f"last_detections: {len(latest.detections)}")
            print(f"coordinate_space: {latest.coordinate_space}")
            report["evidence"].update(
                {
                    "last_capture_ts_ns": latest.capture_ts_ns,
                    "last_detections": len(latest.detections),
                    "coordinate_space": latest.coordinate_space,
                }
            )
        if bool(status.get("terminal_error", False)):
            reason = str(status.get("last_error") or "deepstream_terminal_error")
            print(f"reason: {reason}")
            return _finish_deepstream_smoke_report(args, report, 2, reason)
        if int(status["published_batches"]) <= 0:
            print("reason: deepstream_no_detection_batches_published")
            return _finish_deepstream_smoke_report(
                args,
                report,
                2,
                "deepstream_no_detection_batches_published",
            )
        if int(status.get("tensor_meta_frames", 0)) <= 0:
            print("reason: deepstream_no_tensor_meta_frames")
            return _finish_deepstream_smoke_report(
                args,
                report,
                2,
                "deepstream_no_tensor_meta_frames",
            )
        if int(status.get("postprocess_frames", 0)) <= 0:
            print("reason: deepstream_no_postprocess_frames")
            return _finish_deepstream_smoke_report(
                args,
                report,
                2,
                "deepstream_no_postprocess_frames",
            )
        if observed <= 0:
            print("reason: deepstream_no_detection_batches_observed")
            return _finish_deepstream_smoke_report(
                args,
                report,
                2,
                "deepstream_no_detection_batches_observed",
            )
        timestamp_source = str(status.get("timestamp_source", "") or "")
        if timestamp_source != "gst_clock_base_time_pts":
            reason = f"deepstream_timestamp_source_not_gst_clock_base_time_pts:{timestamp_source}"
            print(f"reason: {reason}")
            return _finish_deepstream_smoke_report(args, report, 2, reason)
        contract_reason = _deepstream_detection_batch_contract_reason(
            latest,
            roi_width=int(args.roi_size),
            roi_height=int(args.roi_size),
        )
        if contract_reason:
            print(f"reason: {contract_reason}")
            return _finish_deepstream_smoke_report(args, report, 2, contract_reason)
        return _finish_deepstream_smoke_report(args, report, 0, "")
    except Exception as exc:
        print("available: False")
        print(f"reason: {exc}")
        return _finish_deepstream_smoke_report(args, report, 2, str(exc))
    finally:
        if started:
            backend.stop()


def _doctor_deepstream_smoke_report(args: argparse.Namespace) -> int:
    report_path = Path(args.report_json).expanduser()
    print(f"report_json: {report_path}")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print("accepted: False")
        print("reason: deepstream_smoke_report_unreadable")
        print(f"detail: {exc}")
        return 2
    failures = _validate_deepstream_smoke_report(report, args=args)
    if failures:
        print("accepted: False")
        print("reason: deepstream_smoke_report_invalid")
        for failure in failures:
            print(f"missing: {failure}")
        return 2
    evidence = report["evidence"]
    print("accepted: True")
    print(f"model_id: {evidence.get('model_id', '')}")
    print(f"engine: {evidence.get('engine', '')}")
    print(f"model_fingerprint: {evidence.get('model_fingerprint', '')}")
    print(f"nvinfer_config_fingerprint: {evidence.get('nvinfer_config_fingerprint', '')}")
    print(f"timestamp_source: {evidence.get('timestamp_source', '')}")
    print(f"tensor_meta_fps: {float(evidence.get('tensor_meta_fps', 0.0)):.2f}")
    print(f"postprocess_fps: {float(evidence.get('postprocess_fps', 0.0)):.2f}")
    print(f"detection_batch_fps: {float(evidence.get('detection_batch_fps', 0.0)):.2f}")
    print(f"last_frame_age_ms: {float(evidence.get('last_frame_age_ms', 0.0)):.2f}")
    print(f"coordinate_space: {evidence.get('coordinate_space', '')}")
    return 0


def _validate_deepstream_smoke_report(
    report: object,
    *,
    args: argparse.Namespace | None = None,
) -> list[str]:
    if not isinstance(report, dict):
        return ["report_object"]
    failures: list[str] = []
    if report.get("check") != "deepstream-smoke":
        failures.append("check")
    if report.get("accepted") is not True:
        failures.append("accepted")
    evidence = report.get("evidence")
    if not isinstance(evidence, dict):
        return failures + ["evidence"]
    required = (
        "dependency_available",
        "model_id",
        "engine",
        "model_fingerprint",
        "nvinfer_config_fingerprint",
        "model_runtime",
        "model_input",
        "model_output",
        "runtime_model_runtime",
        "runtime_model_input",
        "runtime_model_output",
        "pipeline",
        "published_batches",
        "observed_batches",
        "tensor_meta_frames",
        "postprocess_frames",
        "tensor_meta_fps",
        "postprocess_fps",
        "detection_batch_fps",
        "last_frame_id",
        "last_frame_age_ms",
        "last_inference_latency_ms",
        "capture_to_tensor_meta_ms_stats",
        "latency_source",
        "timestamp_source",
        "last_capture_ts_ns",
        "last_detections",
        "coordinate_space",
    )
    for key in required:
        value = evidence.get(key)
        if value is None or value == "":
            failures.append(f"evidence.{key}")
    for key in ("published_batches", "observed_batches", "tensor_meta_frames", "postprocess_frames"):
        try:
            if int(evidence.get(key, 0)) <= 0:
                failures.append(f"evidence.{key}>0")
        except (TypeError, ValueError):
            failures.append(f"evidence.{key}>0")
    for key in ("tensor_meta_fps", "postprocess_fps", "detection_batch_fps"):
        try:
            if float(evidence.get(key, 0.0)) <= 0.0:
                failures.append(f"evidence.{key}>0")
        except (TypeError, ValueError):
            failures.append(f"evidence.{key}>0")
    if evidence.get("dependency_available") is not True:
        failures.append("evidence.dependency_available==true")
    if evidence.get("latency_source") != "capture_to_tensor_meta_done":
        failures.append("evidence.latency_source==capture_to_tensor_meta_done")
    if evidence.get("timestamp_source") != "gst_clock_base_time_pts":
        failures.append("evidence.timestamp_source==gst_clock_base_time_pts")
    if evidence.get("coordinate_space") != "roi":
        failures.append("evidence.coordinate_space==roi")
    _validate_deepstream_model_contract(evidence, failures=failures)
    _validate_deepstream_latency_stats(evidence, failures=failures)
    if args is not None:
        _validate_deepstream_report_thresholds(evidence, args=args, failures=failures)
    return failures


def _validate_deepstream_model_contract(
    evidence: dict[str, object],
    *,
    failures: list[str],
) -> None:
    runtime_required = ("backend", "precision", "batch_size")
    input_required = (
        "name",
        "shape",
        "dtype",
        "layout",
        "color_format",
        "scale_factor",
        "maintain_aspect_ratio",
        "symmetric_padding",
        "width",
        "height",
    )
    output_required = ("name", "shape", "class_count", "class_names")
    for root, required_keys in (
        ("model_runtime", runtime_required),
        ("runtime_model_runtime", runtime_required),
        ("model_input", input_required),
        ("runtime_model_input", input_required[:-2]),
        ("model_output", output_required),
        ("runtime_model_output", output_required),
    ):
        value = evidence.get(root)
        if not isinstance(value, dict):
            failures.append(f"evidence.{root}")
            continue
        for key in required_keys:
            if key not in value:
                failures.append(f"evidence.{root}.{key}")
                continue
            field = value.get(key)
            if field is None or field == "":
                failures.append(f"evidence.{root}.{key}")


def _validate_deepstream_latency_stats(
    evidence: dict[str, object],
    *,
    failures: list[str],
) -> None:
    stats = evidence.get("capture_to_tensor_meta_ms_stats")
    if not isinstance(stats, dict):
        failures.append("evidence.capture_to_tensor_meta_ms_stats")
        return
    for key in ("count", "avg", "p50", "p95", "p99", "max"):
        if key not in stats:
            failures.append(f"evidence.capture_to_tensor_meta_ms_stats.{key}")
            continue
        try:
            value = float(stats.get(key, 0.0))
        except (TypeError, ValueError):
            failures.append(f"evidence.capture_to_tensor_meta_ms_stats.{key}")
            continue
        if key == "count":
            if int(value) <= 0:
                failures.append("evidence.capture_to_tensor_meta_ms_stats.count>0")
        elif value < 0.0:
            failures.append(f"evidence.capture_to_tensor_meta_ms_stats.{key}>=0")


def _validate_deepstream_report_thresholds(
    evidence: dict[str, object],
    *,
    args: argparse.Namespace,
    failures: list[str],
) -> None:
    thresholds = (
        ("tensor_meta_fps", "min_tensor_meta_fps", "min"),
        ("postprocess_fps", "min_postprocess_fps", "min"),
        ("detection_batch_fps", "min_detection_batch_fps", "min"),
        ("last_frame_age_ms", "max_frame_age_ms", "max"),
    )
    for evidence_key, arg_key, mode in thresholds:
        threshold = float(getattr(args, arg_key, 0.0) or 0.0)
        if threshold <= 0.0:
            continue
        try:
            value = float(evidence.get(evidence_key, 0.0))
        except (TypeError, ValueError):
            failures.append(f"evidence.{evidence_key}_{mode}_{threshold:g}")
            continue
        if mode == "min" and value < threshold:
            failures.append(f"evidence.{evidence_key}>={threshold:g}")
        if mode == "max" and value > threshold:
            failures.append(f"evidence.{evidence_key}<={threshold:g}")


def _deepstream_smoke_report(args: argparse.Namespace) -> dict[str, object]:
    return {
        "schema_version": 1,
        "check": "deepstream-smoke",
        "started_at": datetime.now(UTC).isoformat(),
        "accepted": False,
        "reason": "not_finished",
        "parameters": {
            "manifest": str(Path(args.manifest).expanduser()),
            "nvinfer_config": str(Path(args.nvinfer_config).expanduser()),
            "device": args.device,
            "capture_width": args.capture_width,
            "capture_height": args.capture_height,
            "fps": args.fps,
            "roi_left": args.roi_left,
            "roi_top": args.roi_top,
            "roi_size": args.roi_size,
            "io_mode": args.io_mode,
            "batched_push_timeout_us": args.batched_push_timeout_us,
            "seconds": args.seconds,
            "poll_interval": args.poll_interval,
        },
        "evidence": {},
    }


def _finish_deepstream_smoke_report(
    args: argparse.Namespace,
    report: dict[str, object],
    exit_code: int,
    reason: str,
) -> int:
    report["accepted"] = int(exit_code) == 0
    report["reason"] = reason
    report["exit_code"] = int(exit_code)
    _finish_and_write_report(getattr(args, "report_json", None), report)
    return int(exit_code)


def _deepstream_detection_batch_contract_reason(
    batch: object | None,
    *,
    roi_width: int,
    roi_height: int,
) -> str:
    if batch is None:
        return ""
    coordinate_space = str(getattr(batch, "coordinate_space", "") or "")
    if coordinate_space != "roi":
        return f"deepstream_detection_batch_coordinate_space_not_roi:{coordinate_space}"
    width = max(1.0, float(roi_width))
    height = max(1.0, float(roi_height))
    detections = list(getattr(batch, "detections", []) or [])
    for index, detection in enumerate(detections):
        score = float(getattr(detection, "score", float("nan")))
        x1 = float(getattr(detection, "x1", float("nan")))
        y1 = float(getattr(detection, "y1", float("nan")))
        x2 = float(getattr(detection, "x2", float("nan")))
        y2 = float(getattr(detection, "y2", float("nan")))
        cx = float(getattr(detection, "cx", float("nan")))
        cy = float(getattr(detection, "cy", float("nan")))
        values = (score, x1, y1, x2, y2, cx, cy)
        if not all(math.isfinite(value) for value in values):
            return f"deepstream_detection_non_finite:{index}"
        if score < 0.0 or score > 1.0:
            return f"deepstream_detection_score_out_of_range:{index}:{score:.3f}"
        if x2 <= x1 or y2 <= y1:
            return f"deepstream_detection_invalid_box:{index}"
        if cx < 0.0 or cx > width or cy < 0.0 or cy > height:
            return f"deepstream_detection_center_out_of_roi:{index}:{cx:.1f},{cy:.1f}"
    return ""


def _print_zero_copy_report_evidence(report: dict[str, object]) -> None:
    phases = report.get("phases")
    phases = phases if isinstance(phases, dict) else {}
    build = phases.get("build")
    smoke = phases.get("smoke")
    build_stdout = str(build.get("stdout") or "") if isinstance(build, dict) else ""
    smoke_stdout = str(smoke.get("stdout") or "") if isinstance(smoke, dict) else ""
    build_fields = _stdout_fields(build_stdout)
    smoke_fields = _stdout_fields(smoke_stdout)
    summary_fields = (
        "build_library",
        "smoke_library",
        "status_zero_copy",
        "status_memory_space",
        "capture_backend",
        "capture_device",
        "capture_profile",
        "frame_id",
        "capture_ts_ns",
        "capture_ts_source",
        "source_ts_kind",
        "resource_memory",
        "resource_source",
        "resource_size",
        "resource_format",
        "dmabuf_fd",
        "preprocess_backend",
        "preprocess_zero_copy",
        "smoke_release_token",
        "smoke_device_owner_release",
        "tensorrt_preprocess_backend",
        "tensorrt_input_location",
        "tensorrt_release_token",
        "tensorrt_device_owner_release",
    )
    print("evidence: " + ", ".join(summary_fields))
    if "library" in build_fields:
        print(f"evidence.build_library: {build_fields['library']}")
    if "library" in smoke_fields:
        print(f"evidence.smoke_library: {smoke_fields['library']}")
    for key in (
        "status_backend",
        "status_zero_copy",
        "status_memory_space",
    ):
        if key in build_fields:
            print(f"evidence.{key}: {build_fields[key]}")
    for key in (
        "capture_backend",
        "capture_device",
        "capture_profile",
        "frame_id",
        "capture_ts_ns",
        "capture_ts_source",
        "source_ts_ns",
        "source_ts_kind",
        "frame_image_present",
        "frame_size",
        "frame_format",
        "source_size",
        "roi_offset",
        "resource_kind",
        "resource_memory",
        "resource_source",
        "resource_size",
        "resource_format",
        "dmabuf_fd",
        "prepared_frame_id",
        "prepared_capture_ts_ns",
        "prepared_resource_kind",
        "prepared_resource_memory",
        "prepared_resource_source",
        "prepared_resource_size",
        "prepared_resource_format",
        "prepared_dmabuf_fd",
        "input_shape",
        "input_dtype",
        "bridge_available",
        "bridge_native_ready",
        "preprocess_backend",
        "preprocess_input_mode",
        "preprocess_resource_kind",
        "preprocess_resource_memory",
        "preprocess_location",
        "preprocess_zero_copy",
        "tensor_nbytes",
        "tensor_shape",
        "tensor_dtype",
        "smoke_release_token",
        "smoke_device_owner_release",
        "tensorrt_engine",
        "tensorrt_frame_id",
        "tensorrt_capture_ts_ns",
        "tensorrt_last_input_mode",
        "tensorrt_last_input_frame_id",
        "tensorrt_last_input_capture_ts_ns",
        "tensorrt_last_input_resource_kind",
        "tensorrt_last_input_resource_memory",
        "tensorrt_last_input_resource_source",
        "tensorrt_last_input_resource_size",
        "tensorrt_last_input_resource_format",
        "tensorrt_last_input_dmabuf_fd",
        "tensorrt_preprocess_backend",
        "tensorrt_preprocess_location",
        "tensorrt_preprocess_zero_copy",
        "tensorrt_input_location",
        "tensorrt_output_name",
        "tensorrt_output_shape",
        "tensorrt_output_dtype",
        "tensorrt_decoded_detections",
        "tensorrt_execute_enqueue_ms",
        "tensorrt_d2h_enqueue_ms",
        "tensorrt_release_token",
        "tensorrt_device_owner_release",
    ):
        if key in smoke_fields:
            print(f"evidence.{key}: {smoke_fields[key]}")


def _validate_jetson_zero_copy_report(
    report: object,
    *,
    require_tensorrt_engine: bool = False,
) -> list[str]:
    failures: list[str] = []
    if not isinstance(report, dict):
        return ["report must be a JSON object"]
    if report.get("schema_version") != 1:
        failures.append("schema_version must be 1")
    if report.get("check") != "jetson-zero-copy":
        failures.append("check must be jetson-zero-copy")
    if report.get("accepted") is not True:
        failures.append("accepted must be true")
    else:
        if str(report.get("reason") or "").strip():
            failures.append("reason must be empty when accepted is true")
        if report.get("validation_failures") != []:
            failures.append(
                "validation_failures must be an empty list when accepted is true"
            )
    _validate_zero_copy_report_timestamps(report, failures)
    parameters = report.get("parameters")
    if not isinstance(parameters, dict):
        failures.append("parameters object is required")
        parameters = {}
    if not str(parameters.get("library") or "").strip():
        failures.append("parameters.library is required")
    failures.extend(
        _validate_zero_copy_parameters(
            parameters,
            require_tensorrt_engine=require_tensorrt_engine,
        )
    )
    phases = report.get("phases")
    if not isinstance(phases, dict):
        failures.append("phases object is required")
        phases = {}
    build = phases.get("build")
    smoke = phases.get("smoke")
    failures.extend(
        _validate_zero_copy_phase(
            build,
            phase_name="build",
            expected_name="jetson-native-build",
            required_stdout=(
                "production_source:",
                "configure_command:",
                "library:",
                "status_available: True",
                "status_backend:",
                "status_zero_copy: True",
                "status_memory_space: cuda_device",
            ),
            forbidden_backend_tokens=("reference", "scaffold"),
        )
    )
    failures.extend(
        _validate_zero_copy_phase(
            smoke,
            phase_name="smoke",
            expected_name="jetson-native-smoke",
            required_stdout=(
                "capture_available: True",
                "library:",
                "capture_backend: gst-resource:",
                "capture_device:",
                "capture_profile:",
                "frame_id:",
                "capture_ts_ns:",
                "capture_ts_source: userspace_monotonic_receive",
                "source_ts_ns:",
                "source_ts_kind:",
                "frame_image_present: False",
                "frame_size:",
                "frame_format: NV12",
                "source_size:",
                "roi_offset:",
                "userspace_process_ms:",
                "resource_kind: gstreamer_sample",
                "resource_memory: nvmm",
                "resource_source: appsink",
                "resource_size:",
                "resource_format: NV12",
                "dmabuf_fd:",
                "prepared_mode: gpu_buffer",
                "prepared_frame_id:",
                "prepared_capture_ts_ns:",
                "prepared_resource_kind: gstreamer_sample",
                "prepared_resource_memory: nvmm",
                "prepared_resource_source: appsink",
                "prepared_resource_size:",
                "prepared_resource_format: NV12",
                "prepared_dmabuf_fd:",
                "input_shape:",
                "input_dtype:",
                "bridge_available: True",
                "bridge_native_ready: True",
                "preprocess_backend:",
                "preprocess_input_mode: gpu_buffer",
                "preprocess_resource_kind: gstreamer_sample",
                "preprocess_resource_memory: nvmm",
                "preprocess_location: device",
                "preprocess_zero_copy: True",
                "smoke_release_token:",
                "smoke_device_owner_release: released",
                "tensor_device_ptr:",
                "tensor_nbytes:",
                "tensor_shape:",
                "tensor_dtype:",
            ),
        )
    )
    if isinstance(build, dict):
        failures.extend(_validate_zero_copy_build_values(build, parameters))
    if isinstance(smoke, dict):
        failures.extend(_validate_zero_copy_smoke_values(smoke, parameters))
        failures.extend(_validate_zero_copy_tensorrt_values(smoke, parameters))
    return failures


def _validate_zero_copy_report_timestamps(
    report: dict[str, object],
    failures: list[str],
) -> None:
    started_text = str(report.get("started_at") or "").strip()
    finished_text = str(report.get("finished_at") or "").strip()
    if not started_text:
        failures.append("started_at is required")
        return
    if not finished_text:
        failures.append("finished_at is required")
        return
    started = _parse_report_datetime(started_text, field="started_at", failures=failures)
    finished = _parse_report_datetime(
        finished_text,
        field="finished_at",
        failures=failures,
    )
    if started is None or finished is None:
        return
    if finished < started:
        failures.append("finished_at must be greater than or equal to started_at")


def _parse_report_datetime(
    value: str,
    *,
    field: str,
    failures: list[str],
) -> datetime | None:
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        failures.append(f"{field} must be ISO-8601 datetime")
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        failures.append(f"{field} must include timezone")
        return None
    return parsed


def _validate_zero_copy_phase(
    phase: object,
    *,
    phase_name: str,
    expected_name: str,
    required_stdout: tuple[str, ...],
    forbidden_backend_tokens: tuple[str, ...] = (),
) -> list[str]:
    failures: list[str] = []
    if not isinstance(phase, dict):
        return [f"phases.{phase_name} object is required"]
    if phase.get("name") != expected_name:
        failures.append(f"phases.{phase_name}.name must be {expected_name}")
    if phase.get("exit_code") != 0:
        failures.append(f"phases.{phase_name}.exit_code must be 0")
    if phase.get("accepted") is not True:
        failures.append(f"phases.{phase_name}.accepted must be true")
    phase_object_reason = str(phase.get("reason") or "").strip()
    if phase_object_reason:
        failures.append(
            f"phases.{phase_name}.reason must be empty when accepted is true"
        )
    stdout = str(phase.get("stdout") or "")
    required_keys = tuple(
        needle.split(":", 1)[0].strip()
        for needle in required_stdout
        if ":" in needle and needle.split(":", 1)[0].strip()
    )
    for key in _stdout_duplicate_keys(stdout, required_keys):
        failures.append(f"phases.{phase_name}.stdout {key} must appear exactly once")
    if _stdout_duplicate_keys(stdout, ("reason",)):
        failures.append(f"phases.{phase_name}.stdout reason must appear at most once")
    reason_values: list[str] = []
    for line in stdout.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key.strip() == "reason":
            reason_values.append(value.strip())
    phase_reason = next((value for value in reason_values if value), "")
    if phase_reason:
        failures.append(
            f"phases.{phase_name}.stdout reason must be empty for accepted "
            f"phase evidence, got {phase_reason}"
        )
    for needle in required_stdout:
        if needle not in stdout:
            failures.append(f"phases.{phase_name}.stdout missing {needle}")
    for line in stdout.splitlines():
        if not line.lower().startswith("status_backend:"):
            continue
        backend = line.split(":", 1)[1].strip().lower()
        for token in forbidden_backend_tokens:
            if token in backend:
                failures.append(
                    f"phases.{phase_name}.stdout backend must not contain {token}"
                )
    return failures


def _stdout_fields(stdout: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in stdout.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key:
            fields[key] = value.strip()
    return fields


def _stdout_duplicate_keys(stdout: str, keys: tuple[str, ...]) -> tuple[str, ...]:
    counts: dict[str, int] = {}
    key_set = set(keys)
    for line in stdout.splitlines():
        if ":" not in line:
            continue
        key = line.split(":", 1)[0].strip()
        if key in key_set:
            counts[key] = counts.get(key, 0) + 1
    return tuple(key for key in keys if counts.get(key, 0) > 1)


def _validate_zero_copy_build_values(
    phase: dict[str, object],
    parameters: dict[str, object] | None = None,
) -> list[str]:
    failures: list[str] = []
    parameters = parameters or {}
    stdout = str(phase.get("stdout") or "")
    for key in _stdout_duplicate_keys(
        stdout,
        (
            "production_source",
            "configure_command",
            "library",
            "status_available",
            "status_backend",
            "status_zero_copy",
            "status_memory_space",
        ),
    ):
        failures.append(f"phases.build.stdout {key} must appear exactly once")
    fields = _stdout_fields(stdout)
    production_source = fields.get("production_source", "")
    if not production_source:
        failures.append("phases.build.stdout production_source value is required")
    else:
        configured_source = str(parameters.get("source") or "").strip()
        if configured_source and configured_source != "bundled":
            if production_source != configured_source:
                failures.append(
                    "phases.build.stdout production_source must match "
                    f"parameters.source, got {production_source}"
                )
        elif configured_source == "bundled":
            bundled_source = _bundled_jetson_cuda_source_path()
            if not _same_filesystem_path(production_source, bundled_source):
                failures.append(
                    "phases.build.stdout production_source must be the bundled "
                    f"Jetson CUDA source ({bundled_source}), got {production_source}"
                )
        elif not production_source.endswith(
            "novasight_jetson_preprocess_native_jetson_cuda.cu"
        ):
            failures.append(
                "phases.build.stdout production_source must be the bundled "
                f"Jetson CUDA source, got {production_source}"
            )
    configure_command = fields.get("configure_command", "")
    if not configure_command:
        failures.append("phases.build.stdout configure_command value is required")
    else:
        if "-DNOVASIGHT_JETSON_PREPROCESS_IMPL=jetson" not in configure_command:
            failures.append(
                "phases.build.stdout configure_command must set "
                "NOVASIGHT_JETSON_PREPROCESS_IMPL=jetson"
            )
        if production_source and (
            f"-DNOVASIGHT_JETSON_PREPROCESS_PRODUCTION_SOURCE={production_source}"
            not in configure_command
        ):
            failures.append(
                "phases.build.stdout configure_command must include the same "
                "NOVASIGHT_JETSON_PREPROCESS_PRODUCTION_SOURCE as production_source"
            )
    library = fields.get("library", "")
    if not library:
        failures.append("phases.build.stdout library value is required")
    else:
        configured_library = str(parameters.get("library") or "").strip()
        if configured_library and library != configured_library:
            failures.append(
                "phases.build.stdout library must match parameters.library, "
                f"got {library}"
            )
    backend = fields.get("status_backend", "")
    if not backend:
        failures.append("phases.build.stdout status_backend value is required")
    elif not backend.endswith(":jetson_cuda"):
        failures.append(
            f"phases.build.stdout status_backend must end with :jetson_cuda, got {backend}"
        )
    _require_stdout_value(fields, "status_available", "True", failures, "build")
    _require_stdout_value(fields, "status_zero_copy", "True", failures, "build")
    _require_stdout_value(
        fields, "status_memory_space", "cuda_device", failures, "build"
    )
    return failures


def _bundled_jetson_cuda_source_path() -> str:
    try:
        import novasight_jetson_preprocess_native as native_backend

        native_dir = Path(native_backend.__file__).resolve().parent / "native"
    except Exception:
        native_dir = (
            Path(__file__).resolve().parent.parent
            / "novasight_jetson_preprocess_native"
            / "native"
        )
    return str(
        (
            native_dir
            / "src/jetson/novasight_jetson_preprocess_native_jetson_cuda.cu"
        ).resolve(strict=False)
    )


def _same_filesystem_path(left: str, right: str) -> bool:
    try:
        left_path = Path(left).expanduser().resolve(strict=False)
        right_path = Path(right).expanduser().resolve(strict=False)
    except Exception:
        return left == right
    return left_path == right_path


def _validate_zero_copy_parameters(
    parameters: dict[str, object],
    *,
    require_tensorrt_engine: bool = False,
) -> list[str]:
    failures: list[str] = []
    for key in ("source", "device", "pixel_format", "input_shape", "dtype"):
        if not str(parameters.get(key) or "").strip():
            failures.append(f"parameters.{key} is required for zero-copy acceptance")
    for key in ("width", "height", "fps", "roi_size"):
        value = _parameter_int(parameters, key)
        if value is None:
            failures.append(
                f"parameters.{key} must be a positive integer for zero-copy acceptance"
            )
    if require_tensorrt_engine and not str(parameters.get("tensorrt_engine") or "").strip():
        failures.append(
            "parameters.tensorrt_engine is required when --require-tensorrt-engine is set"
        )
    return failures


def _validate_zero_copy_smoke_values(
    phase: dict[str, object],
    parameters: dict[str, object],
) -> list[str]:
    failures: list[str] = []
    stdout = str(phase.get("stdout") or "")
    for key in _stdout_duplicate_keys(
        stdout,
        (
            "available",
            "capture_available",
            "library",
            "capture_backend",
            "capture_device",
            "capture_profile",
            "frame_id",
            "capture_ts_ns",
            "capture_ts_source",
            "source_ts_ns",
            "source_ts_kind",
            "frame_image_present",
            "userspace_process_ms",
            "frame_size",
            "frame_format",
            "source_size",
            "roi_offset",
            "resource_kind",
            "resource_memory",
            "resource_source",
            "resource_size",
            "resource_format",
            "dmabuf_fd",
            "prepared_mode",
            "prepared_frame_id",
            "prepared_capture_ts_ns",
            "prepared_resource_kind",
            "prepared_resource_memory",
            "prepared_resource_source",
            "prepared_resource_size",
            "prepared_resource_format",
            "prepared_dmabuf_fd",
            "input_shape",
            "input_dtype",
            "bridge_available",
            "bridge_native_ready",
            "preprocess_backend",
            "preprocess_input_mode",
            "preprocess_resource_kind",
            "preprocess_resource_memory",
            "preprocess_location",
            "preprocess_zero_copy",
            "smoke_release_token",
            "smoke_device_owner_release",
            "tensor_device_ptr",
            "tensor_nbytes",
            "tensor_shape",
            "tensor_dtype",
        ),
    ):
        failures.append(f"phases.smoke.stdout {key} must appear exactly once")
    fields = _stdout_fields(stdout)
    library = fields.get("library", "")
    if not library:
        failures.append("phases.smoke.stdout library value is required")
    else:
        configured_library = str(parameters.get("library") or "").strip()
        if configured_library and library != configured_library:
            failures.append(
                "phases.smoke.stdout library must match parameters.library, "
                f"got {library}"
            )
    _require_stdout_value(fields, "available", "True", failures, "smoke")
    _require_stdout_value(fields, "capture_available", "True", failures, "smoke")
    _require_positive_int(fields, "frame_id", failures, "smoke")
    _require_positive_int(fields, "capture_ts_ns", failures, "smoke")
    _require_stdout_value(
        fields,
        "capture_ts_source",
        "userspace_monotonic_receive",
        failures,
        "smoke",
    )
    _validate_source_timestamp_diagnostics(fields, failures)
    _validate_zero_copy_roi_offset(fields, parameters, failures)
    capture_backend = fields.get("capture_backend", "")
    if not capture_backend.startswith("gst-resource:"):
        failures.append(
            f"phases.smoke.stdout capture_backend must start with gst-resource:, got {capture_backend or '<missing>'}"
        )
    _validate_capture_device(fields, parameters, failures)
    _require_stdout_value(fields, "frame_format", "NV12", failures, "smoke")
    _require_stdout_value(fields, "frame_image_present", "False", failures, "smoke")
    _require_non_negative_float(fields, "userspace_process_ms", failures, "smoke")
    _require_stdout_value(fields, "resource_kind", "gstreamer_sample", failures, "smoke")
    _require_stdout_value(fields, "resource_memory", "nvmm", failures, "smoke")
    _require_stdout_value(fields, "resource_source", "appsink", failures, "smoke")
    _require_stdout_value(fields, "resource_format", "NV12", failures, "smoke")
    _require_stdout_value(fields, "prepared_mode", "gpu_buffer", failures, "smoke")
    _require_positive_int(fields, "prepared_frame_id", failures, "smoke")
    _require_positive_int(fields, "prepared_capture_ts_ns", failures, "smoke")
    _require_stdout_value(
        fields, "prepared_resource_kind", "gstreamer_sample", failures, "smoke"
    )
    _require_stdout_value(
        fields, "prepared_resource_memory", "nvmm", failures, "smoke"
    )
    _require_stdout_value(
        fields, "prepared_resource_source", "appsink", failures, "smoke"
    )
    _require_stdout_value(fields, "prepared_resource_format", "NV12", failures, "smoke")
    _require_stdout_value(fields, "bridge_available", "True", failures, "smoke")
    _require_stdout_value(fields, "bridge_native_ready", "True", failures, "smoke")
    preprocess_backend = fields.get("preprocess_backend", "")
    if not preprocess_backend:
        failures.append("phases.smoke.stdout preprocess_backend value is required")
    else:
        backend_lower = preprocess_backend.lower()
        for token in ("reference", "scaffold"):
            if token in backend_lower:
                failures.append(
                    f"phases.smoke.stdout preprocess_backend must not contain {token}"
                )
    _require_stdout_value(fields, "preprocess_location", "device", failures, "smoke")
    _require_stdout_value(fields, "preprocess_zero_copy", "True", failures, "smoke")
    _require_stdout_value(fields, "preprocess_input_mode", "gpu_buffer", failures, "smoke")
    _require_stdout_value(
        fields, "preprocess_resource_kind", "gstreamer_sample", failures, "smoke"
    )
    _require_stdout_value(fields, "preprocess_resource_memory", "nvmm", failures, "smoke")
    _require_positive_int(fields, "smoke_release_token", failures, "smoke")
    _require_stdout_value(
        fields, "smoke_device_owner_release", "released", failures, "smoke"
    )
    _require_non_negative_int(fields, "dmabuf_fd", failures, "smoke")
    _require_non_negative_int(fields, "prepared_dmabuf_fd", failures, "smoke")
    _require_positive_int(fields, "tensor_device_ptr", failures, "smoke")
    _require_positive_int(fields, "tensor_nbytes", failures, "smoke")
    expected_tensor_shape = _expected_tensor_shape(parameters, failures)
    expected_tensor_nbytes = (
        expected_tensor_shape.nbytes if expected_tensor_shape is not None else None
    )
    actual_tensor_nbytes = _stdout_int(fields, "tensor_nbytes")
    if (
        expected_tensor_nbytes is not None
        and actual_tensor_nbytes is not None
        and actual_tensor_nbytes != expected_tensor_nbytes
    ):
        failures.append(
            "phases.smoke.stdout tensor_nbytes must match input_shape/dtype "
            f"({expected_tensor_nbytes}), got {actual_tensor_nbytes}"
        )
    if expected_tensor_shape is not None:
        expected_tensor_shape_text = str(
            (
                expected_tensor_shape.batch,
                expected_tensor_shape.channels,
                expected_tensor_shape.height,
                expected_tensor_shape.width,
            )
        )
        _require_stdout_value(
            fields,
            "tensor_shape",
            expected_tensor_shape_text,
            failures,
            "smoke",
        )
        _require_stdout_value(
            fields,
            "tensor_dtype",
            expected_tensor_shape.dtype,
            failures,
            "smoke",
        )
    dmabuf_fd = fields.get("dmabuf_fd")
    prepared_dmabuf_fd = fields.get("prepared_dmabuf_fd")
    if dmabuf_fd and prepared_dmabuf_fd and dmabuf_fd != prepared_dmabuf_fd:
        failures.append(
            "phases.smoke.stdout prepared_dmabuf_fd must match dmabuf_fd"
        )
    frame_id = fields.get("frame_id")
    prepared_frame_id = fields.get("prepared_frame_id")
    if frame_id and prepared_frame_id and frame_id != prepared_frame_id:
        failures.append("phases.smoke.stdout prepared_frame_id must match frame_id")
    capture_ts_ns = fields.get("capture_ts_ns")
    prepared_capture_ts_ns = fields.get("prepared_capture_ts_ns")
    if capture_ts_ns and prepared_capture_ts_ns and capture_ts_ns != prepared_capture_ts_ns:
        failures.append(
            "phases.smoke.stdout prepared_capture_ts_ns must match capture_ts_ns"
        )
    resource_kind = fields.get("resource_kind")
    prepared_resource_kind = fields.get("prepared_resource_kind")
    if (
        resource_kind
        and prepared_resource_kind
        and resource_kind != prepared_resource_kind
    ):
        failures.append(
            "phases.smoke.stdout prepared_resource_kind must match resource_kind"
        )
    resource_source = fields.get("resource_source")
    prepared_resource_source = fields.get("prepared_resource_source")
    if (
        resource_source
        and prepared_resource_source
        and resource_source != prepared_resource_source
    ):
        failures.append(
            "phases.smoke.stdout prepared_resource_source must match resource_source"
        )
    resource_size = fields.get("resource_size")
    prepared_resource_size = fields.get("prepared_resource_size")
    if resource_size and prepared_resource_size and resource_size != prepared_resource_size:
        failures.append(
            "phases.smoke.stdout prepared_resource_size must match resource_size"
        )
    resource_format = fields.get("resource_format")
    prepared_resource_format = fields.get("prepared_resource_format")
    if (
        resource_format
        and prepared_resource_format
        and resource_format != prepared_resource_format
    ):
        failures.append(
            "phases.smoke.stdout prepared_resource_format must match resource_format"
        )
    prepared_mode = fields.get("prepared_mode")
    preprocess_input_mode = fields.get("preprocess_input_mode")
    if (
        prepared_mode
        and preprocess_input_mode
        and prepared_mode != preprocess_input_mode
    ):
        failures.append(
            "phases.smoke.stdout preprocess_input_mode must match prepared_mode"
        )
    preprocess_resource_kind = fields.get("preprocess_resource_kind")
    if (
        prepared_resource_kind
        and preprocess_resource_kind
        and prepared_resource_kind != preprocess_resource_kind
    ):
        failures.append(
            "phases.smoke.stdout preprocess_resource_kind must match prepared_resource_kind"
        )
    prepared_resource_memory = fields.get("prepared_resource_memory")
    preprocess_resource_memory = fields.get("preprocess_resource_memory")
    if (
        prepared_resource_memory
        and preprocess_resource_memory
        and prepared_resource_memory != preprocess_resource_memory
    ):
        failures.append(
            "phases.smoke.stdout preprocess_resource_memory must match prepared_resource_memory"
        )
    expected_shape = str(expected_tensor_shape) if expected_tensor_shape is not None else ""
    if expected_shape:
        _require_stdout_value(fields, "input_shape", expected_shape, failures, "smoke")
    elif not fields.get("input_shape"):
        failures.append("phases.smoke.stdout input_shape value is required")
    expected_dtype = expected_tensor_shape.dtype if expected_tensor_shape is not None else ""
    if expected_dtype:
        _require_stdout_value(fields, "input_dtype", expected_dtype, failures, "smoke")
    elif not fields.get("input_dtype"):
        failures.append("phases.smoke.stdout input_dtype value is required")
    expected_source_size = _expected_size(parameters, "width", "height")
    if expected_source_size:
        _require_stdout_value(
            fields, "source_size", expected_source_size, failures, "smoke"
        )
    elif not fields.get("source_size"):
        failures.append("phases.smoke.stdout source_size value is required")
    expected_roi_size = _expected_square(parameters, "roi_size")
    if expected_roi_size:
        _require_stdout_value(
            fields, "frame_size", expected_roi_size, failures, "smoke"
        )
        _require_stdout_value(
            fields, "resource_size", expected_roi_size, failures, "smoke"
        )
    elif not fields.get("frame_size"):
        failures.append("phases.smoke.stdout frame_size value is required")
    elif not fields.get("resource_size"):
        failures.append("phases.smoke.stdout resource_size value is required")
    frame_size = fields.get("frame_size", "")
    resource_size = fields.get("resource_size", "")
    if frame_size and resource_size and frame_size != resource_size:
        failures.append("phases.smoke.stdout resource_size must match frame_size")
    _validate_capture_profile(fields, parameters, failures)
    return failures


def _validate_capture_device(
    fields: dict[str, str],
    parameters: dict[str, object],
    failures: list[str],
) -> None:
    expected = str(parameters.get("device") or "").strip()
    actual = fields.get("capture_device")
    if actual is None:
        failures.append("phases.smoke.stdout missing capture_device:")
        return
    if not actual:
        failures.append("phases.smoke.stdout capture_device value is required")
        return
    if expected and actual != expected:
        failures.append(
            "phases.smoke.stdout capture_device must match parameters.device "
            f"({expected}), got {actual}"
        )


def _validate_zero_copy_roi_offset(
    fields: dict[str, str],
    parameters: dict[str, object],
    failures: list[str],
) -> None:
    expected_x = _parameter_int_allow_zero(parameters, "roi_offset_x")
    expected_y = _parameter_int_allow_zero(parameters, "roi_offset_y")
    if expected_x is None:
        expected_x = 0
    if expected_y is None:
        expected_y = 0
    expected = f"{expected_x},{expected_y}"
    actual = fields.get("roi_offset")
    if actual is None:
        failures.append("phases.smoke.stdout missing roi_offset:")
        return
    parts = actual.split(",")
    if len(parts) != 2:
        failures.append(
            f"phases.smoke.stdout roi_offset must be two integers x,y, got {actual}"
        )
        return
    try:
        actual_x = int(parts[0].strip())
        actual_y = int(parts[1].strip())
    except ValueError:
        failures.append(
            f"phases.smoke.stdout roi_offset must be two integers x,y, got {actual}"
        )
        return
    if f"{actual_x},{actual_y}" != expected:
        failures.append(
            "phases.smoke.stdout roi_offset must match parameters.roi_offset_x/y "
            f"({expected}), got {actual_x},{actual_y}"
        )


def _validate_source_timestamp_diagnostics(
    fields: dict[str, str],
    failures: list[str],
) -> None:
    source_ts_ns = fields.get("source_ts_ns")
    source_ts_kind = fields.get("source_ts_kind")
    if source_ts_ns is None:
        failures.append("phases.smoke.stdout missing source_ts_ns:")
        return
    if source_ts_kind is None:
        failures.append("phases.smoke.stdout missing source_ts_kind:")
        return
    allowed_kinds = {"", "gstreamer_pts", "gstreamer_dts"}
    if source_ts_kind not in allowed_kinds:
        failures.append(
            "phases.smoke.stdout source_ts_kind must be one of "
            "gstreamer_pts, gstreamer_dts, or empty, "
            f"got {source_ts_kind}"
        )
        return
    if source_ts_kind:
        try:
            parsed = int(source_ts_ns)
        except ValueError:
            failures.append(
                "phases.smoke.stdout source_ts_ns must be an integer >= 0 "
                f"when source_ts_kind={source_ts_kind}, got {source_ts_ns}"
            )
            return
        if parsed < 0:
            failures.append(
                "phases.smoke.stdout source_ts_ns must be >= 0 "
                f"when source_ts_kind={source_ts_kind}, got {parsed}"
            )
        return
    if source_ts_ns not in {"", "None", "none", "null"}:
        failures.append(
            "phases.smoke.stdout source_ts_ns must be None/empty when "
            f"source_ts_kind is empty, got {source_ts_ns}"
        )


def _validate_zero_copy_tensorrt_values(
    phase: dict[str, object],
    parameters: dict[str, object],
) -> list[str]:
    failures: list[str] = []
    engine = str(parameters.get("tensorrt_engine") or "").strip()
    if not engine:
        return failures
    stdout = str(phase.get("stdout") or "")
    fields = _stdout_fields(stdout)
    required_keys = (
        "tensorrt_engine",
        "tensorrt_available",
        "frame_id",
        "capture_ts_ns",
        "tensorrt_frame_id",
        "tensorrt_capture_ts_ns",
        "tensorrt_last_input_frame_id",
        "tensorrt_last_input_capture_ts_ns",
        "tensorrt_input_shape",
        "tensorrt_input_dtype",
        "tensorrt_output_name",
        "tensorrt_output_shape",
        "tensorrt_output_dtype",
        "tensorrt_decoded_detections",
        "tensorrt_last_input_mode",
        "tensorrt_last_input_resource_kind",
        "tensorrt_last_input_resource_memory",
        "tensorrt_last_input_resource_source",
        "tensorrt_last_input_resource_size",
        "tensorrt_last_input_resource_format",
        "tensorrt_last_input_dmabuf_fd",
        "tensorrt_preprocess_backend",
        "tensorrt_preprocess_location",
        "tensorrt_preprocess_zero_copy",
        "tensorrt_input_location",
        "tensorrt_execute_enqueue_ms",
        "tensorrt_d2h_enqueue_ms",
        "tensorrt_release_token",
        "tensorrt_device_owner_release",
    )
    for key in required_keys:
        if key not in fields:
            failures.append(f"phases.smoke.stdout missing {key}:")
    for key in _stdout_duplicate_keys(stdout, required_keys):
        failures.append(f"phases.smoke.stdout {key} must appear exactly once")
    if _stdout_duplicate_keys(stdout, ("tensorrt_reason",)):
        failures.append("phases.smoke.stdout tensorrt_reason must appear at most once")
    tensorrt_reason = str(fields.get("tensorrt_reason") or "").strip()
    if tensorrt_reason:
        failures.append(
            "phases.smoke.stdout tensorrt_reason must be empty for accepted "
            f"TensorRT evidence, got {tensorrt_reason}"
        )
    actual_engine = str(fields.get("tensorrt_engine") or "").strip()
    if not actual_engine:
        failures.append("phases.smoke.stdout tensorrt_engine value is required")
    elif actual_engine != engine:
        failures.append(
            "phases.smoke.stdout tensorrt_engine must match parameters.tensorrt_engine"
        )
    frame_id = _stdout_int(fields, "frame_id")
    capture_ts_ns = _stdout_int(fields, "capture_ts_ns")
    tensorrt_frame_id = _stdout_int(fields, "tensorrt_frame_id")
    tensorrt_capture_ts_ns = _stdout_int(fields, "tensorrt_capture_ts_ns")
    tensorrt_last_input_frame_id = _stdout_int(fields, "tensorrt_last_input_frame_id")
    tensorrt_last_input_capture_ts_ns = _stdout_int(
        fields,
        "tensorrt_last_input_capture_ts_ns",
    )
    for key in (
        "frame_id",
        "capture_ts_ns",
        "tensorrt_frame_id",
        "tensorrt_capture_ts_ns",
        "tensorrt_last_input_frame_id",
        "tensorrt_last_input_capture_ts_ns",
    ):
        _require_positive_int(fields, key, failures, "smoke")
    if (
        tensorrt_frame_id is not None
        and frame_id is not None
        and tensorrt_frame_id != frame_id
    ):
        failures.append("phases.smoke.stdout tensorrt_frame_id must match frame_id")
    if (
        tensorrt_capture_ts_ns is not None
        and capture_ts_ns is not None
        and tensorrt_capture_ts_ns != capture_ts_ns
    ):
        failures.append(
            "phases.smoke.stdout tensorrt_capture_ts_ns must match capture_ts_ns"
        )
    if (
        tensorrt_last_input_frame_id is not None
        and frame_id is not None
        and tensorrt_last_input_frame_id != frame_id
    ):
        failures.append(
            "phases.smoke.stdout tensorrt_last_input_frame_id must match frame_id"
        )
    if (
        tensorrt_last_input_capture_ts_ns is not None
        and capture_ts_ns is not None
        and tensorrt_last_input_capture_ts_ns != capture_ts_ns
    ):
        failures.append(
            "phases.smoke.stdout tensorrt_last_input_capture_ts_ns must match capture_ts_ns"
        )
    _require_stdout_value(fields, "tensorrt_available", "True", failures, "smoke")
    _require_stdout_value(fields, "tensorrt_last_input_mode", "gpu_buffer", failures, "smoke")
    _require_stdout_value(
        fields, "tensorrt_last_input_resource_kind", "gstreamer_sample", failures, "smoke"
    )
    _require_stdout_value(
        fields, "tensorrt_last_input_resource_memory", "nvmm", failures, "smoke"
    )
    _require_stdout_value(
        fields, "tensorrt_last_input_resource_source", "appsink", failures, "smoke"
    )
    _require_stdout_value(
        fields, "tensorrt_last_input_resource_format", "NV12", failures, "smoke"
    )
    _require_non_negative_int(
        fields, "tensorrt_last_input_dmabuf_fd", failures, "smoke"
    )
    resource_size = fields.get("resource_size")
    prepared_resource_size = fields.get("prepared_resource_size")
    tensorrt_resource_size = fields.get("tensorrt_last_input_resource_size")
    if (
        tensorrt_resource_size
        and resource_size
        and tensorrt_resource_size != resource_size
    ):
        failures.append(
            "phases.smoke.stdout tensorrt_last_input_resource_size must match resource_size"
        )
    if (
        tensorrt_resource_size
        and prepared_resource_size
        and tensorrt_resource_size != prepared_resource_size
    ):
        failures.append(
            "phases.smoke.stdout tensorrt_last_input_resource_size must match prepared_resource_size"
        )
    resource_format = fields.get("resource_format")
    prepared_resource_format = fields.get("prepared_resource_format")
    tensorrt_resource_format = fields.get("tensorrt_last_input_resource_format")
    if (
        tensorrt_resource_format
        and resource_format
        and tensorrt_resource_format != resource_format
    ):
        failures.append(
            "phases.smoke.stdout tensorrt_last_input_resource_format must match resource_format"
        )
    if (
        tensorrt_resource_format
        and prepared_resource_format
        and tensorrt_resource_format != prepared_resource_format
    ):
        failures.append(
            "phases.smoke.stdout tensorrt_last_input_resource_format must match prepared_resource_format"
        )
    dmabuf_fd = _stdout_int(fields, "dmabuf_fd")
    prepared_dmabuf_fd = _stdout_int(fields, "prepared_dmabuf_fd")
    tensorrt_dmabuf_fd = _stdout_int(fields, "tensorrt_last_input_dmabuf_fd")
    if (
        tensorrt_dmabuf_fd is not None
        and dmabuf_fd is not None
        and tensorrt_dmabuf_fd != dmabuf_fd
    ):
        failures.append(
            "phases.smoke.stdout tensorrt_last_input_dmabuf_fd must match dmabuf_fd"
        )
    if (
        tensorrt_dmabuf_fd is not None
        and prepared_dmabuf_fd is not None
        and tensorrt_dmabuf_fd != prepared_dmabuf_fd
    ):
        failures.append(
            "phases.smoke.stdout tensorrt_last_input_dmabuf_fd must match prepared_dmabuf_fd"
        )
    _require_stdout_value(
        fields, "tensorrt_preprocess_location", "device", failures, "smoke"
    )
    _require_stdout_value(
        fields, "tensorrt_preprocess_zero_copy", "True", failures, "smoke"
    )
    _require_stdout_value(fields, "tensorrt_input_location", "device", failures, "smoke")
    _require_non_negative_int(
        fields, "tensorrt_decoded_detections", failures, "smoke"
    )
    _require_non_negative_float(
        fields, "tensorrt_execute_enqueue_ms", failures, "smoke"
    )
    _require_non_negative_float(
        fields, "tensorrt_d2h_enqueue_ms", failures, "smoke"
    )
    _require_positive_int(fields, "tensorrt_release_token", failures, "smoke")
    _require_stdout_value(
        fields, "tensorrt_device_owner_release", "released", failures, "smoke"
    )
    backend = fields.get("tensorrt_preprocess_backend", "")
    if not backend:
        failures.append(
            "phases.smoke.stdout tensorrt_preprocess_backend value is required"
        )
    else:
        backend_lower = backend.lower()
        for token in ("reference", "scaffold"):
            if token in backend_lower:
                failures.append(
                    f"phases.smoke.stdout tensorrt_preprocess_backend must not contain {token}"
                )
        smoke_backend = fields.get("preprocess_backend", "")
        if not smoke_backend:
            failures.append(
                "phases.smoke.stdout preprocess_backend value is required for TensorRT evidence"
            )
        elif smoke_backend != backend:
            failures.append(
                "phases.smoke.stdout tensorrt_preprocess_backend must match preprocess_backend"
            )
    expected_input_shape = str(parameters.get("input_shape") or "").strip()
    actual_input_shape = fields.get("tensorrt_input_shape", "")
    if expected_input_shape and actual_input_shape and actual_input_shape != expected_input_shape:
        failures.append(
            "phases.smoke.stdout tensorrt_input_shape must match parameters.input_shape"
        )
    expected_input_dtype = _normalize_optional_tensor_dtype(parameters.get("dtype"))
    actual_input_dtype = _normalize_optional_tensor_dtype(fields.get("tensorrt_input_dtype"))
    if expected_input_dtype and actual_input_dtype and actual_input_dtype != expected_input_dtype:
        failures.append(
            "phases.smoke.stdout tensorrt_input_dtype must match parameters.dtype"
        )
    _validate_tensorrt_input_contract(fields, failures)
    _validate_tensorrt_output_contract(fields, failures)
    return failures


def _validate_tensorrt_input_contract(
    fields: dict[str, str],
    failures: list[str],
) -> None:
    shape_text = fields.get("tensorrt_input_shape")
    if shape_text:
        try:
            from novasight.inference.input import parse_tensor_input_shape

            shape = parse_tensor_input_shape(shape_text)
        except Exception as exc:
            failures.append(
                f"phases.smoke.stdout tensorrt_input_shape invalid: {exc}"
            )
        else:
            if shape.batch != 1:
                failures.append(
                    "phases.smoke.stdout tensorrt_input_shape batch must be 1, "
                    f"got {shape.batch}"
                )
            if shape.channels != 3:
                failures.append(
                    "phases.smoke.stdout tensorrt_input_shape channels must be 3 "
                    f"for NV12-to-RGB NCHW zero-copy inference, got {shape.channels}"
                )
    dtype_text = fields.get("tensorrt_input_dtype")
    if dtype_text:
        try:
            from novasight.inference.input import normalize_tensor_dtype

            normalize_tensor_dtype(dtype_text)
        except Exception as exc:
            failures.append(
                f"phases.smoke.stdout tensorrt_input_dtype invalid: {exc}"
            )


def _validate_tensorrt_output_contract(
    fields: dict[str, str],
    failures: list[str],
) -> None:
    output_name = fields.get("tensorrt_output_name", "")
    if not output_name:
        failures.append("phases.smoke.stdout tensorrt_output_name value is required")
    shape_text = fields.get("tensorrt_output_shape", "")
    if not shape_text:
        failures.append("phases.smoke.stdout tensorrt_output_shape value is required")
    else:
        dims = _parse_stdout_shape_dims(shape_text)
        if not dims:
            failures.append(
                f"phases.smoke.stdout tensorrt_output_shape invalid: {shape_text}"
            )
        elif any(dim <= 0 for dim in dims):
            failures.append(
                "phases.smoke.stdout tensorrt_output_shape dimensions must be positive"
            )
    dtype_text = fields.get("tensorrt_output_dtype", "")
    if not dtype_text:
        failures.append("phases.smoke.stdout tensorrt_output_dtype value is required")
    else:
        try:
            from novasight.inference.input import normalize_tensor_dtype

            normalize_tensor_dtype(dtype_text)
        except Exception as exc:
            failures.append(
                f"phases.smoke.stdout tensorrt_output_dtype invalid: {exc}"
            )


def _require_stdout_value(
    fields: dict[str, str],
    key: str,
    expected: str,
    failures: list[str],
    phase_name: str,
) -> None:
    actual = fields.get(key)
    if actual is None:
        failures.append(f"phases.{phase_name}.stdout missing {key}:")
    elif actual != expected:
        failures.append(
            f"phases.{phase_name}.stdout {key} must be {expected}, got {actual}"
        )


def _require_non_negative_int(
    fields: dict[str, str],
    key: str,
    failures: list[str],
    phase_name: str,
) -> None:
    _require_int_at_least(fields, key, 0, failures, phase_name)


def _require_positive_int(
    fields: dict[str, str],
    key: str,
    failures: list[str],
    phase_name: str,
) -> None:
    _require_int_at_least(fields, key, 1, failures, phase_name)


def _require_int_at_least(
    fields: dict[str, str],
    key: str,
    minimum: int,
    failures: list[str],
    phase_name: str,
) -> None:
    text = fields.get(key)
    if text is None:
        failures.append(f"phases.{phase_name}.stdout missing {key}:")
        return
    try:
        value = int(text)
    except ValueError:
        failures.append(
            f"phases.{phase_name}.stdout {key} must be an integer >= {minimum}, got {text}"
        )
        return
    if value < minimum:
        failures.append(
            f"phases.{phase_name}.stdout {key} must be >= {minimum}, got {value}"
        )


def _require_non_negative_float(
    fields: dict[str, str],
    key: str,
    failures: list[str],
    phase_name: str,
) -> None:
    text = fields.get(key)
    if text is None:
        failures.append(f"phases.{phase_name}.stdout missing {key}:")
        return
    try:
        value = float(text)
    except ValueError:
        failures.append(
            f"phases.{phase_name}.stdout {key} must be a float >= 0, got {text}"
        )
        return
    if value < 0:
        failures.append(
            f"phases.{phase_name}.stdout {key} must be >= 0, got {value}"
        )


def _stdout_int(fields: dict[str, str], key: str) -> int | None:
    text = fields.get(key)
    if text is None:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _stdout_non_negative_int_value(value: object) -> int | None:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _stdout_non_negative_float_value(value: object) -> float | None:
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _parse_stdout_shape_dims(shape_text: str) -> tuple[int, ...]:
    normalized = (
        shape_text.strip()
        .replace("(", "")
        .replace(")", "")
        .replace("[", "")
        .replace("]", "")
        .replace(",", "x")
        .replace(" ", "")
    )
    if not normalized:
        return ()
    try:
        return tuple(int(part) for part in normalized.split("x") if part)
    except ValueError:
        return ()


def _expected_size(
    parameters: dict[str, object],
    width_key: str,
    height_key: str,
) -> str:
    width = _parameter_int(parameters, width_key)
    height = _parameter_int(parameters, height_key)
    if width is None or height is None:
        return ""
    return f"{width}x{height}"


def _expected_square(parameters: dict[str, object], key: str) -> str:
    size = _parameter_int(parameters, key)
    if size is None:
        return ""
    return f"{size}x{size}"


def _parameter_int(parameters: dict[str, object], key: str) -> int | None:
    value = parameters.get(key)
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _parameter_int_allow_zero(parameters: dict[str, object], key: str) -> int | None:
    value = parameters.get(key)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _expected_tensor_shape(
    parameters: dict[str, object],
    failures: list[str],
) -> object | None:
    shape_text = str(parameters.get("input_shape") or "").strip()
    dtype_text = str(parameters.get("dtype") or "").strip()
    if not shape_text or not dtype_text:
        return None
    try:
        from novasight.inference.input import (
            TensorInputShape,
            normalize_tensor_dtype,
            parse_tensor_input_shape,
        )

        parsed = parse_tensor_input_shape(shape_text)
        return TensorInputShape(
            batch=parsed.batch,
            channels=parsed.channels,
            height=parsed.height,
            width=parsed.width,
            dtype=normalize_tensor_dtype(dtype_text),
        )
    except Exception as exc:
        failures.append(f"parameters.input_shape/dtype invalid: {exc}")
        return None


def _validate_capture_profile(
    fields: dict[str, str],
    parameters: dict[str, object],
    failures: list[str],
) -> None:
    profile = fields.get("capture_profile", "")
    if not profile:
        failures.append("phases.smoke.stdout capture_profile value is required")
        return
    match = _CAPTURE_PROFILE_RE.match(profile)
    if not match:
        failures.append(
            "phases.smoke.stdout capture_profile must be formatted as "
            f"<format> <width>x<height>@<fps>, got {profile}"
        )
        return
    pixel_format = str(parameters.get("pixel_format") or "").strip()
    actual_format = match.group("pixel_format")
    if pixel_format and actual_format.upper() != pixel_format.upper():
        failures.append(
            f"phases.smoke.stdout capture_profile must start with {pixel_format}, got {profile}"
        )
    expected_source_size = _expected_size(parameters, "width", "height")
    actual_source_size = f"{match.group('width')}x{match.group('height')}"
    if expected_source_size and actual_source_size != expected_source_size:
        failures.append(
            f"phases.smoke.stdout capture_profile must include {expected_source_size}, got {profile}"
        )
    fps = _parameter_int(parameters, "fps")
    actual_fps = int(match.group("fps"))
    if fps is not None and actual_fps != fps:
        failures.append(
            f"phases.smoke.stdout capture_profile must include @{fps}, got {profile}"
        )


def _jetson_native_status_is_accepted(status: dict[str, object]) -> bool:
    backend = str(status.get("backend") or "").lower()
    return (
        bool(status.get("available"))
        and status.get("zero_copy") is True
        and status.get("memory_space") == "cuda_device"
        and "reference" not in backend
        and "scaffold" not in backend
    )


def _doctor_jetson_native_smoke(args: argparse.Namespace, cfg: object) -> int:
    from novasight.capture.service import CaptureService
    from novasight.inference.input import normalize_tensor_dtype, prepare_tensor_input
    from novasight.inference.jetson import JetsonGpuResourcePreprocessor
    from novasight.inference.preprocess import DeviceTensor, TensorPreprocessError, prepare_tensor

    library_path = Path(args.library).expanduser() if args.library else None
    if library_path is not None and not library_path.is_file():
        print(f"library: {library_path}")
        print("available: False")
        print("reason: jetson_native_library_missing")
        return 2
    if library_path is not None:
        print(f"library: {library_path}")

    shape = _doctor_tensor_input_shape(
        args.input_shape,
        dtype=args.dtype,
        roi_size=int(args.roi_size or cfg.roi.size),
    )
    cfg.capture.memory = "nvmm"
    service = CaptureService(
        config=cfg.capture,
        roi_size=int(args.roi_size or cfg.roi.size),
        roi_offset_x=int(args.roi_offset_x if args.roi_offset_x is not None else cfg.roi.offset_x),
        roi_offset_y=int(args.roi_offset_y if args.roi_offset_y is not None else cfg.roi.offset_y),
    )
    previous_library = None
    if library_path is not None:
        import novasight_jetson_preprocess_native as native_backend

        previous_library = os.environ.get(native_backend.LIBRARY_ENV)
        os.environ[native_backend.LIBRARY_ENV] = str(library_path)
        _reset_jetson_native_caches()
    try:
        requested_device = args.device or cfg.capture.device
        state = service.configure(
            requested_device,
            preference=args.preference,
            pixel_format=args.pixel_format,
            width=args.width,
            height=args.height,
            fps=args.fps,
        )
        print(f"capture_available: {state.available}")
        print(f"capture_backend: {state.backend}")
        capture_device = str(getattr(state, "device", requested_device) or "")
        print(f"capture_device: {capture_device}")
        if state.profile is not None:
            profile = state.profile
            print(
                "capture_profile: "
                f"{profile.pixel_format} {profile.width}x{profile.height}@{profile.fps}"
            )
        if not state.available:
            print("available: False")
            if state.last_error:
                print(f"reason: {state.last_error}")
            return 2
        capture_contract_failures: list[str] = []
        capture_backend = str(getattr(state, "backend", "") or "")
        if not capture_backend.startswith("gst-resource:"):
            capture_contract_failures.append(
                f"capture_backend must start with gst-resource:, got {capture_backend or '<missing>'}"
            )
        if capture_device != str(requested_device):
            capture_contract_failures.append(
                f"capture_device must match requested device {requested_device}, got {capture_device or '<missing>'}"
            )
        profile = state.profile
        if profile is None:
            capture_contract_failures.append("capture_profile is required")
        else:
            profile_format = str(getattr(profile, "pixel_format", "") or "").upper()
            requested_format = str(args.pixel_format or "").upper()
            if requested_format and profile_format != requested_format:
                capture_contract_failures.append(
                    f"capture_profile pixel_format must be {requested_format}, got {profile_format or '<missing>'}"
                )
            for attr, requested_value in (
                ("width", args.width),
                ("height", args.height),
                ("fps", args.fps),
            ):
                if requested_value is None:
                    continue
                actual_value = int(getattr(profile, attr, 0) or 0)
                if actual_value != int(requested_value):
                    capture_contract_failures.append(
                        f"capture_profile {attr} must be {int(requested_value)}, got {actual_value}"
                    )
        if capture_contract_failures:
            print("available: False")
            print("reason: jetson_native_smoke_capture_contract_mismatch")
            for failure in capture_contract_failures:
                print(f"capture_contract_failure: {failure}")
            return 2

        frame = service.wait_preview_frame(
            after_frame_id=0,
            timeout_s=max(float(args.timeout), 0.0),
        )
        if frame is None:
            print("available: False")
            print("reason: jetson_native_smoke_frame_timeout")
            return 2
        resource = frame.frame_resource
        resource_kind = str(getattr(resource, "kind", "") or "")
        resource_memory = str(getattr(resource, "memory", "") or "")
        resource_source = str(getattr(resource, "source", "") or "")
        resource_width = getattr(resource, "width", None)
        resource_height = getattr(resource, "height", None)
        resource_format = str(getattr(resource, "pixel_format", "") or "").upper()
        resource_dmabuf_fd = getattr(resource, "dmabuf_fd", None)
        profile = state.profile
        expected_capture_width = int(getattr(profile, "width", args.width or 0) or 0)
        expected_capture_height = int(getattr(profile, "height", args.height or 0) or 0)
        expected_roi_size = int(args.roi_size or cfg.roi.size or 0)
        expected_frame_width = expected_roi_size or expected_capture_width
        expected_frame_height = expected_roi_size or expected_capture_height
        print(f"frame_id: {frame.frame_id}")
        print(f"capture_ts_ns: {frame.capture_ts_ns}")
        print(f"capture_ts_source: {frame.capture_ts_source}")
        print(f"source_ts_ns: {frame.source_ts_ns}")
        print(f"source_ts_kind: {frame.source_ts_kind}")
        print(f"frame_image_present: {frame.image is not None}")
        print(f"userspace_process_ms: {frame.userspace_process_ms}")
        print(f"frame_size: {frame.width}x{frame.height}")
        print(f"frame_format: {frame.pixel_format}")
        print(f"source_size: {frame.capture_width}x{frame.capture_height}")
        print(f"roi_offset: {frame.roi_x},{frame.roi_y}")
        print(f"resource_kind: {resource_kind}")
        print(f"resource_memory: {resource_memory}")
        print(f"resource_source: {resource_source}")
        print(f"resource_size: {resource_width}x{resource_height}")
        print(f"resource_format: {resource_format}")
        print(f"dmabuf_fd: {resource_dmabuf_fd}")
        frame_contract_failures: list[str] = []
        try:
            frame_id_value = int(frame.frame_id)
        except (TypeError, ValueError):
            frame_id_value = 0
        if frame_id_value <= 0:
            frame_contract_failures.append(
                f"frame_id must be >= 1, got {frame.frame_id}"
            )
        try:
            capture_ts_value = int(frame.capture_ts_ns)
        except (TypeError, ValueError):
            capture_ts_value = 0
        if capture_ts_value <= 0:
            frame_contract_failures.append(
                f"capture_ts_ns must be >= 1, got {frame.capture_ts_ns}"
            )
        capture_ts_source = str(getattr(frame, "capture_ts_source", "") or "")
        if capture_ts_source != "userspace_monotonic_receive":
            frame_contract_failures.append(
                "capture_ts_source must be userspace_monotonic_receive, "
                f"got {capture_ts_source}"
            )
        source_ts_kind = str(getattr(frame, "source_ts_kind", "") or "")
        source_ts_ns = getattr(frame, "source_ts_ns", None)
        if source_ts_kind not in {"", "gstreamer_pts", "gstreamer_dts"}:
            frame_contract_failures.append(
                "source_ts_kind must be gstreamer_pts, gstreamer_dts, or empty, "
                f"got {source_ts_kind}"
            )
        elif source_ts_kind:
            try:
                parsed_source_ts_ns = int(source_ts_ns)
            except (TypeError, ValueError):
                parsed_source_ts_ns = -1
            if parsed_source_ts_ns < 0:
                frame_contract_failures.append(
                    "source_ts_ns must be >= 0 when source_ts_kind is set, "
                    f"got {source_ts_ns}"
                )
        elif source_ts_ns not in {None, "", "None", "none", "null"}:
            frame_contract_failures.append(
                "source_ts_ns must be None/empty when source_ts_kind is empty, "
                f"got {source_ts_ns}"
            )
        try:
            userspace_process_ms = float(frame.userspace_process_ms)
        except (TypeError, ValueError):
            userspace_process_ms = -1.0
        if userspace_process_ms < 0.0:
            frame_contract_failures.append(
                "userspace_process_ms must be >= 0, "
                f"got {frame.userspace_process_ms}"
            )
        if frame.image is not None:
            frame_contract_failures.append("frame_image_present must be False")
        if str(frame.pixel_format or "").upper() != "NV12":
            frame_contract_failures.append(
                f"frame_format must be NV12, got {frame.pixel_format}"
            )
        if expected_capture_width > 0 and int(frame.capture_width) != expected_capture_width:
            frame_contract_failures.append(
                "source_width must match capture profile "
                f"({expected_capture_width}), got {frame.capture_width}"
            )
        if expected_capture_height > 0 and int(frame.capture_height) != expected_capture_height:
            frame_contract_failures.append(
                "source_height must match capture profile "
                f"({expected_capture_height}), got {frame.capture_height}"
            )
        if expected_frame_width > 0 and int(frame.width) != expected_frame_width:
            frame_contract_failures.append(
                "frame_width must match ROI/output contract "
                f"({expected_frame_width}), got {frame.width}"
            )
        if expected_frame_height > 0 and int(frame.height) != expected_frame_height:
            frame_contract_failures.append(
                "frame_height must match ROI/output contract "
                f"({expected_frame_height}), got {frame.height}"
            )
        if frame_contract_failures:
            print("available: False")
            print("reason: jetson_native_smoke_frame_contract_mismatch")
            for failure in frame_contract_failures:
                print(f"frame_contract_failure: {failure}")
            return 2
        if resource is None or not resource.gpu_accessible:
            print("available: False")
            print("reason: jetson_native_smoke_non_gpu_resource")
            return 2
        resource_contract_failures: list[str] = []
        if resource_kind != "gstreamer_sample":
            resource_contract_failures.append(
                f"resource_kind must be gstreamer_sample, got {resource_kind or '<missing>'}"
            )
        if resource_memory != "nvmm":
            resource_contract_failures.append(
                f"resource_memory must be nvmm, got {resource_memory or '<missing>'}"
            )
        if resource_source != "appsink":
            resource_contract_failures.append(
                f"resource_source must be appsink, got {resource_source or '<missing>'}"
            )
        try:
            resource_width_value = int(resource_width)
        except (TypeError, ValueError):
            resource_width_value = 0
        if resource_width_value != int(frame.width):
            resource_contract_failures.append(
                f"resource_width must match frame width {frame.width}, got {resource_width}"
            )
        try:
            resource_height_value = int(resource_height)
        except (TypeError, ValueError):
            resource_height_value = 0
        if resource_height_value != int(frame.height):
            resource_contract_failures.append(
                f"resource_height must match frame height {frame.height}, got {resource_height}"
            )
        if resource_format != "NV12":
            resource_contract_failures.append(
                f"resource_format must be NV12, got {resource_format or '<missing>'}"
            )
        if resource_contract_failures:
            print("available: False")
            print("reason: jetson_native_smoke_resource_contract_mismatch")
            for failure in resource_contract_failures:
                print(f"resource_contract_failure: {failure}")
            return 2
        if resource_dmabuf_fd is None:
            print("available: False")
            print("reason: jetson_native_smoke_missing_dmabuf_fd")
            return 2

        preprocessor_kwargs = {"module_name": args.module} if args.module else {}
        preprocessor = JetsonGpuResourcePreprocessor(**preprocessor_kwargs)
        status = preprocessor.status()
        print(f"bridge_available: {status.get('available')}")
        print(f"bridge_module: {status.get('module')}")
        print(f"bridge_native_ready: {status.get('native_ready')}")
        if status.get("reason"):
            print(f"bridge_reason: {status.get('reason')}")
        if status.get("detail"):
            print(f"bridge_detail: {status.get('detail')}")
        if status.get("available") is not True or status.get("native_ready") is not True:
            print("available: False")
            print("reason: jetson_native_smoke_bridge_not_ready")
            return 2

        prepared = prepare_tensor_input(frame, shape)
        print(f"prepared_mode: {prepared.mode}")
        print(f"prepared_frame_id: {prepared.frame_id}")
        print(f"prepared_capture_ts_ns: {prepared.capture_ts_ns}")
        print(f"prepared_resource_kind: {prepared.resource_kind}")
        print(f"prepared_resource_memory: {prepared.resource_memory}")
        print(f"prepared_resource_source: {prepared.resource_source}")
        prepared_resource_width = getattr(prepared, "resource_width", 0)
        prepared_resource_height = getattr(prepared, "resource_height", 0)
        prepared_resource_format = str(
            getattr(prepared, "resource_pixel_format", "") or ""
        ).upper()
        print(f"prepared_resource_size: {prepared_resource_width}x{prepared_resource_height}")
        print(f"prepared_resource_format: {prepared_resource_format}")
        print(f"prepared_dmabuf_fd: {prepared.dmabuf_fd}")
        prepared_contract_failures: list[str] = []
        if prepared.mode != "gpu_buffer":
            prepared_contract_failures.append(
                f"prepared_mode must be gpu_buffer, got {prepared.mode}"
            )
        if int(prepared.frame_id) != int(frame.frame_id):
            prepared_contract_failures.append(
                f"prepared_frame_id must match frame_id {frame.frame_id}, got {prepared.frame_id}"
            )
        if int(prepared.capture_ts_ns) != int(frame.capture_ts_ns):
            prepared_contract_failures.append(
                "prepared_capture_ts_ns must match capture_ts_ns "
                f"{frame.capture_ts_ns}, got {prepared.capture_ts_ns}"
            )
        if str(prepared.pixel_format or "").upper() != "NV12":
            prepared_contract_failures.append(
                f"prepared_pixel_format must be NV12, got {prepared.pixel_format}"
            )
        if int(prepared.width) != int(frame.width):
            prepared_contract_failures.append(
                f"prepared_width must match frame width {frame.width}, got {prepared.width}"
            )
        if int(prepared.height) != int(frame.height):
            prepared_contract_failures.append(
                f"prepared_height must match frame height {frame.height}, got {prepared.height}"
            )
        if int(prepared.source_width) != int(frame.capture_width):
            prepared_contract_failures.append(
                "prepared_source_width must match frame source width "
                f"{frame.capture_width}, got {prepared.source_width}"
            )
        if int(prepared.source_height) != int(frame.capture_height):
            prepared_contract_failures.append(
                "prepared_source_height must match frame source height "
                f"{frame.capture_height}, got {prepared.source_height}"
            )
        if int(prepared.offset_x) != int(frame.roi_x):
            prepared_contract_failures.append(
                f"prepared_offset_x must match roi_x {frame.roi_x}, got {prepared.offset_x}"
            )
        if int(prepared.offset_y) != int(frame.roi_y):
            prepared_contract_failures.append(
                f"prepared_offset_y must match roi_y {frame.roi_y}, got {prepared.offset_y}"
            )
        if prepared.resource_kind != resource_kind:
            prepared_contract_failures.append(
                "prepared_resource_kind must match frame resource_kind "
                f"{resource_kind}, got {prepared.resource_kind}"
            )
        if prepared.resource_memory != resource_memory:
            prepared_contract_failures.append(
                "prepared_resource_memory must match frame resource_memory "
                f"{resource_memory}, got {prepared.resource_memory}"
            )
        if prepared.resource_source != resource_source:
            prepared_contract_failures.append(
                "prepared_resource_source must match frame resource_source "
                f"{resource_source}, got {prepared.resource_source}"
            )
        try:
            prepared_resource_width_value = int(prepared_resource_width)
        except (TypeError, ValueError):
            prepared_resource_width_value = 0
        if prepared_resource_width_value != int(resource_width_value):
            prepared_contract_failures.append(
                "prepared_resource_width must match frame resource_width "
                f"{resource_width_value}, got {prepared_resource_width}"
            )
        try:
            prepared_resource_height_value = int(prepared_resource_height)
        except (TypeError, ValueError):
            prepared_resource_height_value = 0
        if prepared_resource_height_value != int(resource_height_value):
            prepared_contract_failures.append(
                "prepared_resource_height must match frame resource_height "
                f"{resource_height_value}, got {prepared_resource_height}"
            )
        if prepared_resource_format != resource_format:
            prepared_contract_failures.append(
                "prepared_resource_format must match frame resource_format "
                f"{resource_format}, got {prepared_resource_format or '<missing>'}"
            )
        if prepared.dmabuf_fd != resource_dmabuf_fd:
            prepared_contract_failures.append(
                "prepared_dmabuf_fd must match frame dmabuf_fd "
                f"{resource_dmabuf_fd}, got {prepared.dmabuf_fd}"
            )
        if prepared_contract_failures:
            print("available: False")
            print("reason: jetson_native_smoke_prepared_contract_mismatch")
            for failure in prepared_contract_failures:
                print(f"prepared_contract_failure: {failure}")
            return 2
        print(f"input_shape: {shape}")
        print(f"input_dtype: {shape.dtype}")
        try:
            result = prepare_tensor(
                prepared,
                shape,
                gpu_preprocessor=preprocessor,
            )
        except TensorPreprocessError as exc:
            print("available: False")
            print(f"reason: {exc.reason}")
            print(f"detail: {exc}")
            return 2
        tensor = result.tensor
        if not isinstance(tensor, DeviceTensor):
            print("available: False")
            print("reason: jetson_native_smoke_non_device_tensor")
            return 2
        print(f"preprocess_backend: {result.backend}")
        print(f"preprocess_input_mode: {result.input_mode}")
        print(f"preprocess_resource_kind: {result.resource_kind}")
        print(f"preprocess_resource_memory: {result.resource_memory}")
        print(f"preprocess_location: {result.location}")
        print(f"preprocess_zero_copy: {result.zero_copy}")
        print(f"tensor_device_ptr: {tensor.device_ptr}")
        print(f"tensor_nbytes: {tensor.nbytes}")
        print(f"tensor_shape: {tensor.shape}")
        print(f"tensor_dtype: {tensor.dtype}")
        tensor_contract_failures: list[str] = []
        try:
            tensor_device_ptr = int(tensor.device_ptr)
        except (TypeError, ValueError):
            tensor_device_ptr = 0
        if tensor_device_ptr <= 0:
            tensor_contract_failures.append(
                f"tensor_device_ptr must be >= 1, got {tensor.device_ptr}"
            )
        try:
            tensor_nbytes = int(tensor.nbytes)
        except (TypeError, ValueError):
            tensor_nbytes = 0
        if tensor_nbytes != int(shape.nbytes):
            tensor_contract_failures.append(
                f"tensor_nbytes must be {shape.nbytes}, got {tensor.nbytes}"
            )
        expected_tensor_shape = (
            int(shape.batch),
            int(shape.channels),
            int(shape.height),
            int(shape.width),
        )
        try:
            actual_tensor_shape = tuple(int(item) for item in tuple(tensor.shape))
        except (TypeError, ValueError):
            actual_tensor_shape = ()
        if actual_tensor_shape != expected_tensor_shape:
            tensor_contract_failures.append(
                f"tensor_shape must be {expected_tensor_shape}, got {tensor.shape}"
            )
        try:
            actual_tensor_dtype = normalize_tensor_dtype(tensor.dtype)
        except ValueError:
            actual_tensor_dtype = ""
        if actual_tensor_dtype != shape.dtype:
            tensor_contract_failures.append(
                f"tensor_dtype must be {shape.dtype}, got {tensor.dtype}"
            )
        preprocess_contract_failures: list[str] = []
        if result.input_mode != prepared.mode:
            preprocess_contract_failures.append(
                f"preprocess_input_mode must match prepared_mode {prepared.mode}, got {result.input_mode}"
            )
        if result.resource_kind != prepared.resource_kind:
            preprocess_contract_failures.append(
                "preprocess_resource_kind must match prepared_resource_kind "
                f"{prepared.resource_kind}, got {result.resource_kind}"
            )
        if result.resource_memory != prepared.resource_memory:
            preprocess_contract_failures.append(
                "preprocess_resource_memory must match prepared_resource_memory "
                f"{prepared.resource_memory}, got {result.resource_memory}"
            )
        if result.location != "device":
            preprocess_contract_failures.append(
                f"preprocess_location must be device, got {result.location}"
            )
        if result.zero_copy is not True:
            preprocess_contract_failures.append(
                f"preprocess_zero_copy must be True, got {result.zero_copy}"
            )
        smoke_release_token = _tensor_owner_release_token(tensor)
        print(f"smoke_release_token: {smoke_release_token}")
        smoke_owner_release = _release_tensor_owner(tensor)
        print(f"smoke_device_owner_release: {smoke_owner_release}")
        if smoke_release_token is None:
            print("available: False")
            print("reason: jetson_native_smoke_release_evidence_missing")
            return 2
        if smoke_owner_release != "released":
            print("available: False")
            print("reason: jetson_native_smoke_device_owner_not_released")
            return 2
        if tensor_contract_failures:
            print("available: False")
            print("reason: jetson_native_smoke_tensor_contract_mismatch")
            for failure in tensor_contract_failures:
                print(f"tensor_contract_failure: {failure}")
            return 2
        if preprocess_contract_failures:
            print("available: False")
            print("reason: jetson_native_smoke_preprocess_contract_mismatch")
            for failure in preprocess_contract_failures:
                print(f"preprocess_contract_failure: {failure}")
            return 2
        if not str(result.backend or "").strip():
            print("available: False")
            print("reason: jetson_native_smoke_preprocess_backend_missing")
            return 2
        if _backend_name_is_reference_or_scaffold(result.backend):
            print("available: False")
            print("reason: jetson_native_smoke_non_production_backend")
            return 2
        tensorrt_engine = str(getattr(args, "tensorrt_engine", None) or "").strip()
        if tensorrt_engine:
            result = _doctor_jetson_tensorrt_binding(
                engine_path=Path(tensorrt_engine).expanduser(),
                classes=_doctor_classes(getattr(args, "classes", None)),
                input_shape=str(shape),
                input_dtype=shape.dtype,
                frame=frame,
                preprocessor=preprocessor,
            )
            if result != 0:
                return result
        print("available: True")
        return 0
    finally:
        service.stop("jetson native smoke complete")
        if library_path is not None:
            import novasight_jetson_preprocess_native as native_backend

            if previous_library is None:
                os.environ.pop(native_backend.LIBRARY_ENV, None)
            else:
                os.environ[native_backend.LIBRARY_ENV] = previous_library
            _reset_jetson_native_caches()


def _doctor_tensor_input_shape(
    value: str | None,
    *,
    dtype: str | None,
    roi_size: int,
):
    from novasight.inference.input import TensorInputShape, normalize_tensor_dtype, parse_tensor_input_shape

    shape = parse_tensor_input_shape(value) if value else TensorInputShape(1, 3, roi_size, roi_size)
    normalized_dtype = normalize_tensor_dtype(dtype or shape.dtype)
    return TensorInputShape(
        batch=shape.batch,
        channels=shape.channels,
        height=shape.height,
        width=shape.width,
        dtype=normalized_dtype,
    )


def _doctor_classes(value: str | None) -> list[str]:
    if not value:
        return ["class0"]
    classes = [item.strip() for item in value.split(",") if item.strip()]
    return classes or ["class0"]


def _release_tensor_owner(tensor: object) -> str:
    owner = getattr(tensor, "owner", None)
    if owner is None:
        return "not_present"
    release = getattr(owner, "release", None)
    if not callable(release):
        return "not_callable"
    try:
        release()
    except Exception as exc:
        return f"failed:{exc}"
    return "released"


def _tensor_owner_release_token(tensor: object) -> int | None:
    owner = getattr(tensor, "owner", None)
    if owner is None:
        return None
    for attr in ("release_token", "token", "_token"):
        value = getattr(owner, attr, None)
        if callable(value):
            try:
                value = value()
            except Exception:
                continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def _doctor_jetson_tensorrt_binding(
    *,
    engine_path: Path,
    classes: list[str],
    input_shape: str,
    input_dtype: str | None = None,
    frame: object,
    preprocessor: object,
) -> int:
    print(f"tensorrt_engine: {engine_path}")
    print(f"tensorrt_frame_id: {getattr(frame, 'frame_id', '')}")
    print(f"tensorrt_capture_ts_ns: {getattr(frame, 'capture_ts_ns', '')}")
    if not engine_path.is_file():
        print("available: False")
        print("tensorrt_available: False")
        print("reason: jetson_native_smoke_tensorrt_engine_missing")
        return 2
    from novasight.inference.tensorrt import TensorRtInferenceEngine

    engine = TensorRtInferenceEngine(gpu_preprocessor=preprocessor)  # type: ignore[arg-type]
    try:
        engine.load(engine_path, classes=classes, input_shape=input_shape)
        inference = engine.infer(frame)
        status = engine.status()
        debug = inference.debug if isinstance(inference.debug, dict) else {}
        preprocess = debug.get("preprocess", {}) if isinstance(debug, dict) else {}
        preprocess_debug = preprocess if isinstance(preprocess, dict) else {}
        decode = debug.get("decode", {}) if isinstance(debug, dict) else {}
        decode_debug = decode if isinstance(decode, dict) else {}
        timings = decode_debug.get("timings", {})
        timings_debug = timings if isinstance(timings, dict) else {}
        tensorrt_preprocess_backend = str(
            preprocess_debug.get("preprocess_backend") or ""
        ).strip()
        output_name = str(status.get("output_name") or "")
        output_shape = str(status.get("output_shape") or "")
        output_dtype = str(status.get("output_dtype") or "")
        decoded_detections = debug.get("decoded_detections", "")
        execute_enqueue_ms = timings_debug.get("execute_enqueue_ms", "")
        d2h_enqueue_ms = timings_debug.get("d2h_enqueue_ms", "")
        release_token = timings_debug.get("device_owner_release_token", "")
        device_owner_release = str(timings_debug.get("device_owner_release", ""))
        print(f"tensorrt_available: {inference.available}")
        if inference.reason:
            print(f"tensorrt_reason: {inference.reason}")
        print(f"tensorrt_input_shape: {status.get('input_shape')}")
        print(f"tensorrt_input_dtype: {status.get('input_dtype')}")
        print(f"tensorrt_output_name: {output_name}")
        print(f"tensorrt_output_shape: {output_shape}")
        print(f"tensorrt_output_dtype: {output_dtype}")
        print(f"tensorrt_decoded_detections: {decoded_detections}")
        print(f"tensorrt_last_input_mode: {status.get('last_input_mode')}")
        print(f"tensorrt_last_input_frame_id: {status.get('last_input_frame_id')}")
        print(
            "tensorrt_last_input_capture_ts_ns: "
            f"{status.get('last_input_capture_ts_ns')}"
        )
        print(
            "tensorrt_last_input_resource_kind: "
            f"{status.get('last_input_resource_kind')}"
        )
        print(
            "tensorrt_last_input_resource_memory: "
            f"{status.get('last_input_resource_memory')}"
        )
        print(
            "tensorrt_last_input_resource_source: "
            f"{status.get('last_input_resource_source')}"
        )
        print(
            "tensorrt_last_input_resource_size: "
            f"{status.get('last_input_resource_size')}"
        )
        print(
            "tensorrt_last_input_resource_format: "
            f"{status.get('last_input_resource_format')}"
        )
        print(f"tensorrt_last_input_dmabuf_fd: {status.get('last_input_dmabuf_fd')}")
        print(
            "tensorrt_preprocess_backend: "
            f"{tensorrt_preprocess_backend}"
        )
        print(
            "tensorrt_preprocess_location: "
            f"{preprocess_debug.get('preprocess_location', '')}"
        )
        print(
            "tensorrt_preprocess_zero_copy: "
            f"{preprocess_debug.get('preprocess_zero_copy', '')}"
        )
        print(f"tensorrt_input_location: {timings_debug.get('input_location', '')}")
        print(f"tensorrt_execute_enqueue_ms: {execute_enqueue_ms}")
        print(f"tensorrt_d2h_enqueue_ms: {d2h_enqueue_ms}")
        print(f"tensorrt_release_token: {release_token}")
        print(f"tensorrt_device_owner_release: {device_owner_release}")
        frame_resource = getattr(frame, "frame_resource", None)
        frame_dmabuf_fd = getattr(frame_resource, "dmabuf_fd", None)
        frame_resource_size = (
            f"{getattr(frame_resource, 'width', '')}x{getattr(frame_resource, 'height', '')}"
            if frame_resource is not None
            else ""
        )
        frame_resource_format = str(
            getattr(frame_resource, "pixel_format", "") or ""
        ).upper()
        tensorrt_dmabuf_fd = _stdout_non_negative_int_value(
            status.get("last_input_dmabuf_fd")
        )
        frame_id = _stdout_non_negative_int_value(getattr(frame, "frame_id", None))
        frame_capture_ts_ns = _stdout_non_negative_int_value(
            getattr(frame, "capture_ts_ns", None)
        )
        tensorrt_frame_id = _stdout_non_negative_int_value(
            status.get("last_input_frame_id")
        )
        tensorrt_capture_ts_ns = _stdout_non_negative_int_value(
            status.get("last_input_capture_ts_ns")
        )
        expected_input_shape = str(input_shape or "").strip()
        actual_input_shape = str(status.get("input_shape") or "").strip()
        expected_input_dtype = _normalize_optional_tensor_dtype(input_dtype)
        actual_input_dtype = _normalize_optional_tensor_dtype(status.get("input_dtype"))
        if (
            expected_input_shape
            and actual_input_shape
            and actual_input_shape != expected_input_shape
        ) or (
            expected_input_dtype
            and actual_input_dtype
            and actual_input_dtype != expected_input_dtype
        ):
            print("available: False")
            print("reason: jetson_native_smoke_tensorrt_input_contract_mismatch")
            return 2
        if (
            status.get("last_input_resource_kind") != "gstreamer_sample"
            or status.get("last_input_resource_memory") != "nvmm"
            or status.get("last_input_resource_source") != "appsink"
            or status.get("last_input_resource_size") != frame_resource_size
            or status.get("last_input_resource_format") != frame_resource_format
            or frame_dmabuf_fd is None
            or tensorrt_dmabuf_fd is None
            or int(frame_dmabuf_fd) != tensorrt_dmabuf_fd
            or frame_id is None
            or tensorrt_frame_id is None
            or frame_id != tensorrt_frame_id
            or frame_capture_ts_ns is None
            or tensorrt_capture_ts_ns is None
            or frame_capture_ts_ns != tensorrt_capture_ts_ns
        ):
            print("available: False")
            print("reason: jetson_native_smoke_tensorrt_resource_mismatch")
            return 2
        if not inference.available:
            print("available: False")
            return 2
        if not tensorrt_preprocess_backend:
            print("available: False")
            print("reason: jetson_native_smoke_tensorrt_preprocess_backend_missing")
            return 2
        if (
            status.get("last_input_mode") != "gpu_buffer"
            or status.get("last_input_resource_memory") != "nvmm"
            or preprocess_debug.get("preprocess_location") != "device"
            or preprocess_debug.get("preprocess_zero_copy") is not True
            or timings_debug.get("input_location") != "device"
            or _backend_name_is_reference_or_scaffold(tensorrt_preprocess_backend)
        ):
            print("available: False")
            print("reason: jetson_native_smoke_tensorrt_not_zero_copy")
            return 2
        release_token_value = _stdout_non_negative_int_value(release_token)
        if (
            release_token_value is None
            or release_token_value <= 0
            or device_owner_release != "released"
        ):
            print("available: False")
            print("reason: jetson_native_smoke_tensorrt_release_evidence_missing")
            return 2
        output_shape_dims = _parse_stdout_shape_dims(output_shape)
        output_dtype_valid = False
        if output_dtype:
            try:
                from novasight.inference.input import normalize_tensor_dtype

                normalize_tensor_dtype(output_dtype)
                output_dtype_valid = True
            except Exception:
                output_dtype_valid = False
        if (
            output_name
            and output_shape
            and output_dtype
            and (
                not output_shape_dims
                or any(dim <= 0 for dim in output_shape_dims)
                or not output_dtype_valid
            )
        ):
            print("available: False")
            print("reason: jetson_native_smoke_tensorrt_output_contract_invalid")
            return 2
        if (
            not output_name
            or not output_shape_dims
            or not output_dtype
            or _stdout_non_negative_int_value(decoded_detections) is None
            or _stdout_non_negative_float_value(execute_enqueue_ms) is None
            or _stdout_non_negative_float_value(d2h_enqueue_ms) is None
        ):
            print("available: False")
            print("reason: jetson_native_smoke_tensorrt_output_evidence_missing")
            return 2
        return 0
    except Exception as exc:
        print("available: False")
        print("tensorrt_available: False")
        print(f"reason: {exc}")
        return 2
    finally:
        close = getattr(engine, "close", None)
        if callable(close):
            close()


def _normalize_optional_tensor_dtype(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        from novasight.inference.input import normalize_tensor_dtype

        return normalize_tensor_dtype(text)
    except Exception:
        return text


def _backend_name_is_reference_or_scaffold(value: object) -> bool:
    text = str(value or "").lower()
    return "reference" in text or "scaffold" in text


def _reset_jetson_native_caches() -> None:
    for module_name, reset_name in (
        ("novasight_jetson_preprocess", "_reset_backend_cache"),
        ("novasight_jetson_preprocess_native", "_reset_library_cache"),
    ):
        try:
            module = __import__(module_name)
        except Exception:
            continue
        reset = getattr(module, reset_name, None)
        if callable(reset):
            reset()


def _run_doctor_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    return result
