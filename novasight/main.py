from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import uvicorn

from .api import create_app
from .config import load_runtime_config
from .runtime import configure_logging


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
        print("error: doctor requires a subcommand: camera, kmnet", file=sys.stderr)
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
            flip_dy=cfg.hardware.flip_dy,
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
