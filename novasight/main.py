from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from .api import create_app
from .config import load_runtime_config


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
            state = service.capture_frames(seconds=args.seconds)
            print(f"fps_capture: {state.fps_capture:.2f}")
            print(f"capture_wait_ms: {state.capture_wait_ms:.2f}")
            print(f"frame_period_ms: {state.frame_period_ms:.2f}")
            print(f"frames_dropped: {state.frames_dropped}")
            return 0
        except Exception as exc:
            print(f"reason: {exc}")
            return 2
        finally:
            if service.source is not None:
                service.source.close()
    app = create_app(
        data_dir=Path(args.data_dir),
        config_path=Path(args.config),
        config=cfg,
    )
    uvicorn.run(app, host=cfg.web.host, port=cfg.web.port)
    return 0
