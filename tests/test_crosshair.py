from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
import time

from PIL import Image, ImageDraw
import pytest

from novasight.api import routes_crosshair
from novasight.config import CrosshairConfig, RuntimeConfig, parse_runtime_config
from novasight.control import MouseController, MouseObservation
from novasight.crosshair import ControlReference, CrosshairSystem
from novasight.deepstream.pipeline_builder import (
    DeepStreamPipelineConfig,
    build_deepstream_pipeline,
)
from novasight.deepstream.runtime_pipeline import DeepStreamRuntimePipeline
from novasight.executors import ExecutorRegistry
from novasight.runtime import RuntimeService
from novasight.contracts import FrameContext


def _jpeg(*, offset_x: int = 0, offset_y: int = 0, size: int = 96) -> bytes:
    image = Image.new("RGB", (size, size), (24, 31, 42))
    draw = ImageDraw.Draw(image)
    cx = size // 2 + offset_x
    cy = size // 2 + offset_y
    color = (255, 48, 92)
    draw.line((cx - 13, cy, cx - 4, cy), fill=color, width=2)
    draw.line((cx + 4, cy, cx + 13, cy), fill=color, width=2)
    draw.line((cx, cy - 13, cx, cy - 4), fill=color, width=2)
    draw.line((cx, cy + 4, cx, cy + 13), fill=color, width=2)
    payload = BytesIO()
    image.save(payload, format="JPEG", quality=95)
    return payload.getvalue()


def _config(**values: object) -> CrosshairConfig:
    defaults = {
        "enabled": True,
        "use_for_control": True,
        "search_size": 96,
        "sample_hz": 10,
        "sample_frames": 3,
        "confirm_duration_ms": 100.0,
        "max_age_ms": 250.0,
        "max_offset_px": 20.0,
        "min_similarity": 0.60,
        "max_step_px": 3.0,
    }
    defaults.update(values)
    return CrosshairConfig(**defaults)


def test_crosshair_learns_detects_and_resolves_one_stable_reference(tmp_path) -> None:
    system = CrosshairSystem(_config(), template_path=tmp_path / "template.json")
    base_ns = time.monotonic_ns() - 30_000_000
    for index in range(3):
        system.ingest_jpeg(
            _jpeg(),
            sample_ts_ns=base_ns + index * 10_000_000,
            roi_width=640,
            roi_height=640,
        )

    template = system.learn()
    assert template["point_count"] >= 8
    assert (tmp_path / "template.json").is_file()

    first = system.ingest_jpeg(
        _jpeg(offset_x=4, offset_y=-3),
        sample_ts_ns=base_ns + 100_000_000,
        roi_width=640,
        roi_height=640,
    )
    confirmed = system.ingest_jpeg(
        _jpeg(offset_x=4, offset_y=-3),
        sample_ts_ns=base_ns + 210_000_000,
        roi_width=640,
        roi_height=640,
    )
    reference = system.resolve(
        geometric_x=320.0,
        geometric_y=320.0,
        now_ns=base_ns + 220_000_000,
        geometry_signature="640x640",
    )

    assert first.status == "candidate"
    assert confirmed.status == "confirmed"
    assert confirmed.x == pytest.approx(324.0, abs=1.0)
    assert confirmed.y == pytest.approx(317.0, abs=1.0)
    assert reference.source == "vision_verified"
    assert reference.x == pytest.approx(confirmed.x)
    assert reference.y == pytest.approx(confirmed.y)


def test_crosshair_falls_back_without_template_when_stale_or_geometry_changes(tmp_path) -> None:
    system = CrosshairSystem(_config(), template_path=tmp_path / "template.json")

    missing = system.resolve(
        geometric_x=300.0,
        geometric_y=301.0,
        now_ns=time.monotonic_ns(),
        geometry_signature="640x640",
    )

    assert missing.source == "geometry"
    assert (missing.x, missing.y) == (300.0, 301.0)
    assert missing.reason == "template_unavailable"


def test_crosshair_learning_rejects_stale_or_non_unique_samples() -> None:
    stale = CrosshairSystem(_config(max_age_ms=50.0))
    for index in range(3):
        stale.ingest_jpeg(
            _jpeg(),
            sample_ts_ns=time.monotonic_ns() - 1_000_000_000 + index,
            roi_width=640,
            roi_height=640,
        )
    with pytest.raises(ValueError, match="stale"):
        stale.learn()

    repeated = CrosshairSystem(_config())
    image = Image.new("RGB", (96, 96), (255, 48, 92))
    payload = BytesIO()
    image.save(payload, format="JPEG", quality=95)
    base_ns = time.monotonic_ns()
    for index in range(3):
        repeated.ingest_jpeg(
            payload.getvalue(),
            sample_ts_ns=base_ns + index,
            roi_width=640,
            roi_height=640,
        )
    with pytest.raises(ValueError, match="not unique"):
        repeated.learn()


def test_crosshair_template_persists_and_reloads(tmp_path) -> None:
    path = tmp_path / "template.json"
    first = CrosshairSystem(_config(), template_path=path)
    base_ns = time.monotonic_ns()
    for index in range(3):
        first.ingest_jpeg(
            _jpeg(),
            sample_ts_ns=base_ns + index,
            roi_width=640,
            roi_height=640,
        )
    learned = first.learn()

    second = CrosshairSystem(_config(), template_path=path)

    assert second.status()["template"]["id"] == learned["id"]
    assert second.status()["template"]["point_count"] == learned["point_count"]


def test_crosshair_status_reports_actual_control_readiness() -> None:
    system = CrosshairSystem(_config(confirm_duration_ms=0.0))
    base_ns = time.monotonic_ns() - 20_000_000
    for index in range(3):
        system.ingest_jpeg(
            _jpeg(),
            sample_ts_ns=base_ns + index,
            roi_width=640,
            roi_height=640,
        )
    system.learn()
    system.ingest_jpeg(
        _jpeg(),
        sample_ts_ns=time.monotonic_ns(),
        roi_width=640,
        roi_height=640,
    )

    assert system.status()["control_reference_ready"] is True

    system.update_config(_config(confirm_duration_ms=0.0, use_for_control=False))
    status = system.status()

    assert status["control_reference_ready"] is False
    assert status["control_reference_source"] == "geometry"
    assert status["control_reference_reason"] == "control_disabled"


def test_deepstream_pipeline_adds_independent_crosshair_nvmm_tap(tmp_path) -> None:
    config = DeepStreamPipelineConfig(
        device="/dev/video0",
        capture_width=1920,
        capture_height=1080,
        fps=120,
        roi_left=640,
        roi_top=220,
        roi_width=640,
        roi_height=640,
        model_width=320,
        model_height=320,
        nvinfer_config_path=tmp_path / "nvinfer.ini",
        crosshair_enabled=True,
        crosshair_size=96,
        crosshair_fps=10,
    )

    pipeline = build_deepstream_pipeline(config)

    assert "nvvidconv name=crosshair-crop left=912 right=1008 top=492 bottom=588" in pipeline
    assert "nvjpegenc name=crosshair-encoder" in pipeline
    assert "appsink name=crosshair_sink" in pipeline
    assert "video/x-raw,format=BGR" not in pipeline


def test_mouse_controller_uses_explicit_reference_instead_of_geometric_center() -> None:
    from novasight.control import (
        CALIBRATED_ANGULAR,
        CalibratedAngularControllerConfig,
        MouseControllerConfig,
        SharedOutputConfig,
    )

    controller = MouseController(
        MouseControllerConfig(
            mode=CALIBRATED_ANGULAR,
            calibrated_angular=CalibratedAngularControllerConfig(
                fov_x_deg=90.0,
                counts_per_360_x=1000.0,
                counts_per_360_y=1000.0,
                kp_x=1.0,
                kp_y=1.0,
                kd_x=0.0,
                kd_y=0.0,
                d_ema_alpha=1.0,
                max_angle_step_x_rad=10.0,
                max_angle_step_y_rad=10.0,
            ),
            shared=SharedOutputConfig(
                deadzone_x_px=0.0,
                deadzone_y_px=0.0,
                max_count_slew_x=1000.0,
                max_count_slew_y=1000.0,
                max_budget_counts_x=1000,
                max_budget_counts_y=1000,
            ),
        )
    )
    observation = MouseObservation(
        frame_id=1,
        target_id=1,
        capture_ts_ns=1,
        control_now_ts_ns=2,
        measurement_dt_s=0.01,
        control_width_px=200.0,
        control_height_px=200.0,
        observed_x_px=104.0,
        observed_y_px=97.0,
        predicted_x_px=104.0,
        predicted_y_px=97.0,
        prediction_horizon_s=0.0,
        target_confidence=1.0,
        prediction_confidence=0.0,
        reference_x_px=104.0,
        reference_y_px=97.0,
    )

    command = controller.calculate(observation)

    assert command.dx == 0
    assert command.dy == 0
    assert command.debug["control_reference_x_px"] == 104.0
    assert command.debug["control_reference_y_px"] == 97.0


def test_crosshair_config_parses_and_rejects_invalid_search_geometry() -> None:
    parsed = parse_runtime_config(
        {
            "roi": {"size": 320},
            "crosshair": {
                "enabled": True,
                "search_size": 64,
                "sample_hz": 8,
                "sample_frames": 7,
            },
        }
    )

    assert parsed.crosshair.enabled is True
    assert parsed.crosshair.search_size == 64
    assert parsed.crosshair.sample_frames == 7
    with pytest.raises(ValueError, match="crosshair.search_size"):
        parse_runtime_config({"roi": {"size": 320}, "crosshair": {"search_size": 321}})


def test_crosshair_api_learns_previews_and_clears_template(tmp_path) -> None:
    system = CrosshairSystem(_config(), template_path=tmp_path / "template.json")
    base_ns = time.monotonic_ns()
    for index in range(3):
        system.ingest_jpeg(
            _jpeg(),
            sample_ts_ns=base_ns + index,
            roi_width=640,
            roi_height=640,
        )
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                runtime=SimpleNamespace(crosshair=system),
            )
        )
    )

    learned = routes_crosshair.learn_crosshair(request)
    preview = routes_crosshair.get_crosshair_template_preview(request)
    cleared = routes_crosshair.clear_crosshair_template(request)

    assert learned["learned"] is True
    assert preview.media_type == "image/png"
    assert bytes(preview.body).startswith(b"\x89PNG")
    assert cleared["template"] is None


def test_runtime_freezes_control_reference_for_one_generation_and_frame() -> None:
    class ReferenceSource:
        def __init__(self) -> None:
            self.calls = 0

        def resolve(self, **values: object) -> ControlReference:
            self.calls += 1
            return ControlReference(
                x=320.0 + self.calls,
                y=320.0,
                source="vision_verified",
                confidence=1.0,
                sample_ts_ns=1,
                age_ms=0.0,
                geometry_signature=str(values["geometry_signature"]),
            )

        def update_config(self, _config: CrosshairConfig) -> None:
            return None

        def reset_runtime(self) -> None:
            return None

        def status(self) -> dict[str, object]:
            return {}

    source = ReferenceSource()
    service = RuntimeService(
        RuntimeConfig(),
        models=SimpleNamespace(get_active_deployment=lambda: None),
        executors=ExecutorRegistry.from_config(RuntimeConfig()),
        crosshair=source,  # type: ignore[arg-type]
    )
    first = FrameContext(frame_id=7, generation=1, width=640, height=640)
    restarted = FrameContext(frame_id=7, generation=2, width=640, height=640)

    first_reference = service._control_reference_for_context(first)
    repeated_reference = service._control_reference_for_context(first)
    restarted_reference = service._control_reference_for_context(restarted)

    assert first_reference is repeated_reference
    assert source.calls == 2
    assert restarted_reference.x == 322.0


def test_deepstream_crosshair_observer_delivers_latest_jpeg_to_runtime() -> None:
    delivered: list[dict[str, object]] = []

    class Backend:
        running = True
        pipeline_config = SimpleNamespace(roi_width=640, roi_height=640)
        calls = 0

        def wait_crosshair_jpeg(self, **_values: object) -> tuple[int, bytes, int] | None:
            self.calls += 1
            if self.calls > 1:
                return None
            self.running = False
            return 4, b"jpeg", 123

    runtime = SimpleNamespace(
        process_crosshair_jpeg=lambda payload, **values: delivered.append(
            {"payload": payload, **values}
        )
    )
    pipeline = DeepStreamRuntimePipeline(backend=Backend(), runtime=runtime)  # type: ignore[arg-type]

    pipeline._crosshair_loop()

    assert delivered == [
        {
            "payload": b"jpeg",
            "sample_ts_ns": 123,
            "roi_width": 640,
            "roi_height": 640,
        }
    ]
