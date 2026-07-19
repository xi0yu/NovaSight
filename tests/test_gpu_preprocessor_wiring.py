from __future__ import annotations

import copy
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from novasight.api import create_app, routes_runtime
from novasight.capture.source import (
    CapturedFrame,
    FrameResource,
    _host_frame_resource_from_copy,
    _sample_to_bgr,
    _sample_frame_resource,
)
from novasight.capture.state import CaptureProfile, CaptureRuntimeState
from novasight.config import RuntimeConfig
from novasight.contracts import BBox, Detection, DetectionBatch
from novasight.executors import ExecutorRegistry
from novasight.inference.contracts import InferenceDetection
from novasight.inference.geometry import map_model_detections_to_roi_frame
from novasight.inference.input import PreparedTensorInput, TensorInputShape, prepare_tensor_input
from novasight.inference.jetson import (
    JetsonGpuResourcePreprocessor,
    create_gpu_resource_preprocessor,
)
from novasight.inference.preprocess import (
    DeviceTensor,
    TensorPreprocessResult,
    prepare_host_tensor,
    prepare_tensor,
)
import novasight.inference.tensorrt as tensorrt_module
from novasight.inference.runtime import InferenceRuntime
from novasight.inference.tensorrt import TensorRtInferenceEngine
from novasight.runtime import RuntimeService
from novasight.runtime.reconfigurator import RuntimeReconfigurator


class _FakeEngine:
    engine_id = "fake"

    def __init__(self) -> None:
        self.gpu_preprocessor = object()
        self.confidence_threshold = 0.0
        self.nms_threshold = 0.0

    def available(self) -> bool:
        return True

    def last_reason(self) -> str:
        return ""

    def status(self) -> dict:
        return {"selected": self.engine_id, "available": True, "loaded": False}

    def set_gpu_preprocessor(self, value) -> None:
        self.gpu_preprocessor = value


def test_runtime_start_returns_business_failure_instead_of_http_500(monkeypatch) -> None:
    runtime = SimpleNamespace(
        pipeline=None,
        running=False,
        fatal_error=None,
    )
    app = FastAPI()
    app.include_router(routes_runtime.router)
    app.state.runtime = runtime
    app.state.capture = SimpleNamespace()

    def fail_create(*_args, **_kwargs):
        raise ValueError("model contract mismatch")

    monkeypatch.setattr(routes_runtime, "create_runtime_pipeline", fail_create)

    response = TestClient(app).post("/api/runtime/start")
    payload = response.json()

    assert response.status_code == 200
    assert payload["running"] is False
    assert payload["failed"] is True
    assert payload["accepted"] is False
    assert payload["last_error"] == "model contract mismatch"
    assert runtime.pipeline is None
    assert runtime.fatal_error["message"] == "model contract mismatch"


def test_nvmm_latest_config_creates_jetson_gpu_preprocessor() -> None:
    cfg = RuntimeConfig()
    cfg.inference.enabled = True
    cfg.inference.backend = "nvmm_latest"
    cfg.capture.memory = "nvmm"

    preprocessor = create_gpu_resource_preprocessor(cfg)

    assert isinstance(preprocessor, JetsonGpuResourcePreprocessor)


def test_api_app_wires_gpu_preprocessor_for_default_nvmm_latest(
    tmp_path,
) -> None:
    cfg = RuntimeConfig()
    cfg.inference.backend = "nvmm_latest"
    cfg.capture.memory = "nvmm"

    app = create_app(
        data_dir=tmp_path / "data",
        config_path=tmp_path / "missing.yaml",
        config=cfg,
    )

    assert isinstance(app.state.inference._gpu_preprocessor, JetsonGpuResourcePreprocessor)


def test_gpu_preprocessor_disabled_for_non_nvmm_or_disabled_inference() -> None:
    cfg = RuntimeConfig()
    cfg.inference.enabled = False
    cfg.inference.backend = "nvmm_latest"
    cfg.capture.memory = "nvmm"
    assert create_gpu_resource_preprocessor(cfg) is None

    cfg.inference.enabled = True
    cfg.capture.memory = "cpu"
    assert create_gpu_resource_preprocessor(cfg) is None

    cfg.capture.memory = "nvmm"
    cfg.inference.backend = "native_tensorrt"
    assert create_gpu_resource_preprocessor(cfg) is None


def test_runtime_reconfigurator_rewires_gpu_preprocessor_for_nvmm_latest(
    tmp_path,
) -> None:
    cfg = RuntimeConfig()
    cfg.inference.enabled = True
    cfg.inference.backend = "native_tensorrt"
    cfg.capture.memory = "cpu"
    app = create_app(
        data_dir=tmp_path / "data",
        config_path=tmp_path / "missing.yaml",
        config=cfg,
    )
    assert app.state.inference._gpu_preprocessor is None

    next_cfg = RuntimeConfig()
    next_cfg.inference.enabled = True
    next_cfg.inference.backend = "nvmm_latest"
    next_cfg.capture.memory = "nvmm"

    RuntimeReconfigurator(app).apply(next_cfg)

    assert isinstance(app.state.inference._gpu_preprocessor, JetsonGpuResourcePreprocessor)


def test_runtime_reconfigurator_switches_algorithm_and_executor_under_runtime_lock(
    tmp_path,
    monkeypatch,
) -> None:
    cfg = RuntimeConfig()
    cfg.control.active_algorithm = "dual_phase_atan_robust_predictive_v2"
    app = create_app(
        data_dir=tmp_path / "data",
        config_path=tmp_path / "missing.yaml",
        config=cfg,
    )
    registry = app.state.executors
    original_update = registry.update_runtime_config
    observed_runtime_algorithms: list[str] = []

    def record_update(next_config: RuntimeConfig) -> None:
        observed_runtime_algorithms.append(app.state.runtime.config.control.active_algorithm)
        original_update(next_config)

    monkeypatch.setattr(registry, "update_runtime_config", record_update)
    next_cfg = copy.deepcopy(cfg)
    next_cfg.control.active_algorithm = "universal_saturated"

    RuntimeReconfigurator(app).apply(next_cfg)

    assert observed_runtime_algorithms == ["universal_saturated"]
    assert app.state.runtime.executors is registry
    assert registry.single_command_per_observation is False


def test_runtime_reconfigurator_applies_control_parameters_without_touching_inference(
    tmp_path,
    monkeypatch,
) -> None:
    cfg = RuntimeConfig()
    app = create_app(
        data_dir=tmp_path / "data",
        config_path=tmp_path / "runtime.yaml",
        config=cfg,
    )
    runtime = app.state.runtime
    pipeline = SimpleNamespace(running=True)
    runtime.pipeline = pipeline
    runtime.running = True

    monkeypatch.setattr(
        app.state.inference,
        "configure",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("control-only update must not reconfigure inference")
        ),
    )
    monkeypatch.setattr(
        app.state.inference,
        "unload",
        lambda _reason: (_ for _ in ()).throw(
            AssertionError("control-only update must not unload inference")
        ),
    )
    next_cfg = copy.deepcopy(cfg)
    next_cfg.control.target_fov_radius_px = 320.0

    report = RuntimeReconfigurator(app).apply(next_cfg)

    assert report.applied is True
    assert report.sections[0].impact == "control_hot_update"
    assert runtime.pipeline is pipeline
    assert runtime.running is True
    assert runtime.config.control.target_fov_radius_px == pytest.approx(320.0)


def test_runtime_reconfigurator_hot_updates_power_policy_without_pipeline_rebuild(
    tmp_path,
    monkeypatch,
) -> None:
    cfg = RuntimeConfig()
    app = create_app(
        data_dir=tmp_path / "data",
        config_path=tmp_path / "runtime.yaml",
        config=cfg,
    )
    pipeline = SimpleNamespace(running=True)
    app.state.runtime.pipeline = pipeline
    app.state.runtime.running = True
    monkeypatch.setattr(
        app.state.inference,
        "configure",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("power policy must not reconfigure inference")
        ),
    )
    next_cfg = copy.deepcopy(cfg)
    next_cfg.power_saving.target_host_id = "gaming-pc"
    next_cfg.power_saving.heartbeat_timeout_s = 8.0

    report = RuntimeReconfigurator(app).apply(next_cfg)

    assert report.sections[0].impact == "policy_hot_update"
    assert app.state.runtime.pipeline is pipeline
    assert app.state.runtime.running is True
    assert app.state.runtime.config.power_saving.target_host_id == "gaming-pc"


def test_runtime_reconfigurator_rolls_back_power_policy_when_persistence_fails(
    tmp_path,
    monkeypatch,
) -> None:
    cfg = RuntimeConfig()
    cfg.power_saving.target_host_id = "old-host"
    app = create_app(
        data_dir=tmp_path / "data",
        config_path=tmp_path / "runtime.yaml",
        config=cfg,
    )
    persisted: list[RuntimeConfig] = []
    supervisor_reconfigure_calls: list[dict[str, object]] = []
    original_reconfigure = app.state.runtime_power.reconfigure

    def record_reconfigure(**kwargs):
        supervisor_reconfigure_calls.append(kwargs)
        return original_reconfigure(**kwargs)

    def save_with_first_call_failure(config, _path) -> None:
        persisted.append(config)
        if len(persisted) == 1:
            raise OSError("disk unavailable")

    monkeypatch.setattr(
        "novasight.runtime.reconfigurator.save_runtime_config",
        save_with_first_call_failure,
    )
    monkeypatch.setattr(app.state.runtime_power, "reconfigure", record_reconfigure)
    next_cfg = copy.deepcopy(cfg)
    next_cfg.power_saving.target_host_id = "new-host"

    with pytest.raises(ValueError, match="previous config restored"):
        RuntimeReconfigurator(app).apply(next_cfg)

    assert app.state.config.power_saving.target_host_id == "old-host"
    assert app.state.runtime.config.power_saving.target_host_id == "old-host"
    assert app.state.runtime.config_store.snapshot().power_saving.target_host_id == "old-host"
    assert persisted[-1].power_saving.target_host_id == "old-host"
    assert supervisor_reconfigure_calls == []


def test_runtime_field_update_does_not_build_unused_config_schema(
    tmp_path,
    monkeypatch,
) -> None:
    cfg = RuntimeConfig()
    app = create_app(
        data_dir=tmp_path / "data",
        config_path=tmp_path / "runtime.yaml",
        config=cfg,
    )
    monkeypatch.setattr(
        "novasight.runtime.reconfigurator.runtime_config_schema",
        lambda _config: (_ for _ in ()).throw(
            AssertionError("field updates must not build the full config schema")
        ),
    )

    report = RuntimeReconfigurator(app).apply_field(
        "runtime",
        "freshness_threshold_ms",
        40.0,
    )

    assert report.applied is True
    assert "schema" not in report.asdict()


def test_output_gate_hot_update_preserves_live_target_and_control_modules(
    tmp_path,
) -> None:
    cfg = RuntimeConfig()
    cfg.control.trigger_mode = "always"
    app = create_app(
        data_dir=tmp_path / "data",
        config_path=tmp_path / "runtime.yaml",
        config=cfg,
    )
    runtime = app.state.runtime
    runtime.running = True
    app.state.config_path = None

    def feed(frame_id: int) -> None:
        capture_ts_ns = time.monotonic_ns()
        runtime.process_detection_batch(
            DetectionBatch(
                frame_id=frame_id,
                generation=frame_id,
                capture_ts_ns=capture_ts_ns,
                inference_start_ts_ns=capture_ts_ns + 1_000,
                inference_end_ts_ns=capture_ts_ns + 2_000,
                detections=[
                    Detection(cls=0, score=0.95, x1=330, y1=250, x2=490, y2=568),
                    Detection(cls=1, score=0.94, x1=370, y1=250, x2=430, y2=340),
                ],
                classes=["body", "head"],
                coordinate_space="roi",
            ),
            width=640,
            height=640,
            source_width=1920,
            source_height=1080,
            roi_offset_x=600,
            roi_offset_y=220,
        )

    feed(1)
    feed(2)
    assert runtime.last_control["will_emit"] is True
    active_track_id = runtime.last_target["track_id"]
    algorithms = runtime.control_algorithms
    scheduler = app.state.executors.scheduler

    disabled = RuntimeReconfigurator(app).apply_field(
        "control",
        "output_enabled",
        False,
    )
    enabled = RuntimeReconfigurator(app).apply_field(
        "control",
        "output_enabled",
        True,
    )
    feed(3)

    assert disabled.sections[0].impact == "output_gate_hot_update"
    assert enabled.sections[0].impact == "output_gate_hot_update"
    assert runtime.last_control["will_emit"] is True
    assert runtime.last_target["track_id"] == active_track_id
    assert runtime.control_algorithms is algorithms
    assert app.state.executors.scheduler is scheduler
    assert app.state.executors.status()["output_enabled"] is True


def test_runtime_reconfigurator_applies_aim_mapping_without_full_runtime_rebuild(
    tmp_path,
    monkeypatch,
) -> None:
    cfg = RuntimeConfig()
    app = create_app(
        data_dir=tmp_path / "data",
        config_path=tmp_path / "runtime.yaml",
        config=cfg,
    )
    runtime = app.state.runtime
    resets: list[str] = []
    clears: list[str] = []

    monkeypatch.setattr(runtime.control_algorithms, "reset", lambda: resets.append("algorithm"))
    monkeypatch.setattr(
        runtime,
        "_clear_pending_commands",
        lambda reason: clears.append(reason),
    )
    monkeypatch.setattr(
        runtime.target_selector,
        "reset",
        lambda: (_ for _ in ()).throw(AssertionError("target tracking must be preserved")),
    )
    monkeypatch.setattr(
        app.state.inference,
        "configure",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("inference must not be reconfigured")
        ),
    )
    monkeypatch.setattr(
        app.state.executors,
        "update_runtime_config",
        lambda _config: (_ for _ in ()).throw(AssertionError("executors must not be rebuilt")),
    )
    reconfigurator = RuntimeReconfigurator(app)
    monkeypatch.setattr(
        reconfigurator,
        "_ensure_runtime_pipeline_for_live_capture",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("pipeline readiness must not be checked")
        ),
    )
    next_cfg = copy.deepcopy(cfg)
    next_cfg.control.aim.class_roles = {"default": {"0": "body", "1": "head"}}

    report = reconfigurator.apply(next_cfg)

    assert report.applied is True
    assert report.sections[0].section == "control.aim"
    assert report.sections[0].impact == "aim_mapping_hot_update"
    assert "schema" not in report.asdict()
    assert resets == ["algorithm"]
    assert clears == ["AIM_MAPPING_UPDATED"]
    assert runtime.config.control.aim.class_roles == next_cfg.control.aim.class_roles
    assert (
        runtime.config_store.snapshot().control.aim.class_roles
        == next_cfg.control.aim.class_roles
    )

    selection_cfg = copy.deepcopy(next_cfg)
    selection_cfg.inference.detection_class_profiles["default"] = ["身体", "头部"]
    selection_cfg.inference.detection_class_filter = "0,1"
    selection_cfg.inference.detection_class_priority = "1,0"

    selection_report = reconfigurator.apply(selection_cfg)

    assert selection_report.sections[0].section == "targeting"
    assert selection_report.sections[0].impact == "targeting_hot_update"
    assert resets == ["algorithm"]
    assert clears == ["AIM_MAPPING_UPDATED"]
    assert runtime.config.inference.detection_class_priority == "1,0"

    with ThreadPoolExecutor(max_workers=2) as pool:
        updates = (
            pool.submit(
                reconfigurator.apply_field,
                "inference",
                "detection_class_filter",
                "1",
            ),
            pool.submit(
                reconfigurator.apply_field,
                "control",
                "aim",
                {
                    "role_y_ratios": {"head": 0.18, "body": 0.40, "other": 0.22},
                    "class_roles": {"default": {"0": "body", "1": "head"}},
                },
            ),
        )
        for update in updates:
            update.result()

    assert runtime.config.inference.detection_class_filter == "1"
    assert runtime.config.control.aim.role_y_ratios.head == pytest.approx(0.18)


def test_runtime_reconfigurator_disconnects_old_kmnet_before_hardware_replacement(
    tmp_path,
    monkeypatch,
) -> None:
    cfg = RuntimeConfig()
    app = create_app(
        data_dir=tmp_path / "data",
        config_path=tmp_path / "missing.yaml",
        config=cfg,
    )
    old_kmnet = app.state.executors.executors["kmnet"]
    old_kmnet.connected = True
    events: list[str] = []

    def disconnect_old() -> dict[str, object]:
        events.append("old_disconnected")
        old_kmnet.connected = False
        return {"connected": False}

    old_kmnet.disconnect = disconnect_old
    original_from_config = ExecutorRegistry.from_config

    def build_replacement(config: RuntimeConfig) -> ExecutorRegistry:
        assert events == ["old_disconnected"]
        events.append("replacement_created")
        return original_from_config(config)

    monkeypatch.setattr(ExecutorRegistry, "from_config", build_replacement)
    next_cfg = RuntimeConfig()
    next_cfg.hardware.monitor_port = cfg.hardware.monitor_port + 1

    RuntimeReconfigurator(app).apply(next_cfg)

    assert events == ["old_disconnected", "replacement_created"]
    assert app.state.executors.executors["kmnet"] is not old_kmnet


def test_runtime_reconfigurator_restarts_running_pipeline_after_roi_change_with_stale_runtime_flag(
    tmp_path,
    monkeypatch,
) -> None:
    cfg = RuntimeConfig()
    cfg.source.default = "capture"
    app = create_app(
        data_dir=tmp_path / "data",
        config_path=tmp_path / "missing.yaml",
        config=cfg,
    )
    stopped: list[str] = []

    class ExistingPipeline:
        running = True

        def stop(self) -> None:
            stopped.append("stopped")

    starts: list[str] = []
    waits: list[float] = []

    class ReplacementPipeline:
        running = False

        def start(self) -> None:
            starts.append("started")
            self.running = True
            app.state.runtime.running = True

        def wait_until_ready(self, timeout_s: float) -> bool:
            waits.append(timeout_s)
            return True

        def stop(self) -> None:
            self.running = False

    app.state.runtime.pipeline = ExistingPipeline()
    app.state.runtime.running = False
    replacement = ReplacementPipeline()
    monkeypatch.setattr(
        "novasight.runtime.reconfigurator.create_runtime_pipeline",
        lambda **_kwargs: replacement,
    )

    next_cfg = copy.deepcopy(cfg)
    next_cfg.roi.size = 320

    report = RuntimeReconfigurator(app).apply(next_cfg)

    assert report.applied is True
    assert stopped == ["stopped"]
    assert starts == ["started"]
    assert waits == [5.0]
    assert app.state.capture.roi_size == 320
    assert app.state.runtime.config.roi.size == 320
    assert app.state.runtime.config_store.status()["roi"]["size"] == 320
    assert app.state.runtime.running is True
    assert report.restart_required is False


def test_runtime_reconfigurator_rolls_back_when_pipeline_construction_fails(
    tmp_path,
    monkeypatch,
) -> None:
    cfg = RuntimeConfig()
    cfg.source.default = "capture"
    cfg.inference.enabled = True
    cfg.inference.backend = "deepstream_nvinfer"
    cfg.capture.backend = "deepstream_nvinfer"
    cfg.capture.memory = "nvmm"
    app = create_app(
        data_dir=tmp_path / "data",
        config_path=tmp_path / "runtime.yaml",
        config=cfg,
    )

    class ExistingPipeline:
        running = True

        def stop(self) -> None:
            self.running = False

    app.state.runtime.pipeline = ExistingPipeline()
    app.state.runtime.running = True
    next_cfg = copy.deepcopy(cfg)
    next_cfg.roi.size = 320

    def fail_create(*_args, **_kwargs):
        raise RuntimeError(
            "DeepStream model manifest is missing: data/models/demo/default/model.manifest.json"
        )

    monkeypatch.setattr(
        "novasight.runtime.reconfigurator.create_runtime_pipeline",
        fail_create,
    )

    with pytest.raises(ValueError, match="DeepStream model manifest is missing"):
        RuntimeReconfigurator(app).apply(next_cfg)

    assert app.state.config.roi.size == cfg.roi.size
    assert app.state.runtime.config.roi.size == cfg.roi.size
    assert app.state.runtime.config_store.snapshot().roi.size == cfg.roi.size
    assert app.state.runtime.pipeline is None
    assert app.state.runtime.running is False


def test_runtime_reconfigurator_rolls_back_when_rebuilt_pipeline_never_becomes_ready(
    tmp_path,
    monkeypatch,
) -> None:
    cfg = RuntimeConfig()
    cfg.source.default = "capture"
    app = create_app(
        data_dir=tmp_path / "data",
        config_path=tmp_path / "runtime.yaml",
        config=cfg,
    )

    class ExistingPipeline:
        running = True

        def stop(self) -> None:
            self.running = False

    class CandidatePipeline:
        def __init__(self, *, ready: bool) -> None:
            self.running = False
            self.ready = ready
            self.stopped = False

        def start(self) -> None:
            self.running = True
            app.state.runtime.running = True

        def wait_until_ready(self, _timeout_s: float) -> bool:
            return self.ready

        def status(self) -> dict[str, object]:
            return {"last_error": "first batch missing"}

        def stop(self) -> None:
            self.stopped = True
            self.running = False

    rejected = CandidatePipeline(ready=False)
    restored = CandidatePipeline(ready=True)
    candidates = iter((rejected, restored))
    monkeypatch.setattr(
        "novasight.runtime.reconfigurator.create_runtime_pipeline",
        lambda **_kwargs: next(candidates),
    )
    app.state.runtime.pipeline = ExistingPipeline()
    app.state.runtime.running = True
    next_cfg = copy.deepcopy(cfg)
    next_cfg.roi.size = 320

    with pytest.raises(ValueError, match="previous config restored"):
        RuntimeReconfigurator(app).apply(next_cfg)

    assert rejected.stopped is True
    assert restored.ready is True
    assert app.state.runtime.pipeline is restored
    assert app.state.runtime.config.roi.size == cfg.roi.size
    assert app.state.runtime.running is True


def test_preview_rate_change_requires_a_deepstream_pipeline_rebuild() -> None:
    cfg = RuntimeConfig()
    next_cfg = copy.deepcopy(cfg)
    next_cfg.limits.stream_fps = 15

    assert RuntimeReconfigurator._pipeline_config_changed(cfg, next_cfg) is True
    assert RuntimeReconfigurator._roi_changed(cfg, next_cfg) is False


def test_runtime_service_update_config_rewires_gpu_preprocessor() -> None:
    cfg = RuntimeConfig()
    cfg.inference.enabled = True
    cfg.inference.backend = "native_tensorrt"
    cfg.capture.memory = "cpu"
    inference = InferenceRuntime(_FakeEngine())
    service = RuntimeService(
        cfg,
        models=SimpleNamespace(),
        executors=SimpleNamespace(
            update_runtime_config=lambda _cfg: None,
            status=lambda: {},
        ),
        inference=inference,
    )
    assert inference._gpu_preprocessor is None

    next_cfg = RuntimeConfig()
    next_cfg.inference.enabled = True
    next_cfg.inference.backend = "nvmm_latest"
    next_cfg.capture.memory = "nvmm"

    service.update_config(next_cfg)

    assert isinstance(inference._gpu_preprocessor, JetsonGpuResourcePreprocessor)
    assert inference.engine.gpu_preprocessor is inference._gpu_preprocessor


def test_capture_select_does_not_auto_start_stopped_nvmm_runtime(
    tmp_path,
    monkeypatch,
) -> None:
    cfg = RuntimeConfig()
    cfg.inference.enabled = True
    cfg.inference.backend = "nvmm_latest"
    cfg.capture.memory = "nvmm"
    app = create_app(
        data_dir=tmp_path / "data",
        config_path=tmp_path / "missing.yaml",
        config=cfg,
    )
    profile = CaptureProfile(
        device="/dev/video0",
        pixel_format="MJPG",
        width=2560,
        height=1440,
        fps=120,
        preference="manual",
        selection_reason="manual profile matched device capabilities",
    )
    state = CaptureRuntimeState(
        available=True,
        device="/dev/video0",
        profile=profile,
        backend="gst-resource:nvmm-mjpg-iomode2",
    )
    app.state.capture = SimpleNamespace(
        config=cfg.capture,
        roi_size=cfg.roi.size,
        source=object(),
        session=SimpleNamespace(running=True),
        state=state,
        last_config_error=None,
        configure_profile_only=lambda *args, **kwargs: state,
    )

    def fail_if_started(*args, **kwargs):
        raise AssertionError("capture selection must not auto-start a stopped runtime")

    monkeypatch.setattr("novasight.runtime.pipeline.RuntimePipeline.start", fail_if_started)

    report = RuntimeReconfigurator(app).select_capture(
        device="/dev/video0",
        preference="manual",
        pixel_format="MJPG",
        width=2560,
        height=1440,
        fps=120,
    )

    assert report.applied is True
    assert report.message == "采集配置已应用"
    assert app.state.runtime.running is False


def test_capture_select_reports_runtime_restart_failure_when_pipeline_was_running(
    tmp_path,
    monkeypatch,
) -> None:
    cfg = RuntimeConfig()
    cfg.source.default = "capture"
    cfg.inference.enabled = True
    cfg.capture.width = 1920
    cfg.capture.height = 1080
    cfg.capture.fps = 60
    app = create_app(
        data_dir=tmp_path / "data",
        config_path=tmp_path / "missing.yaml",
        config=cfg,
    )
    profile = CaptureProfile(
        device="/dev/video0",
        pixel_format="MJPG",
        width=2560,
        height=1440,
        fps=120,
        preference="manual",
        selection_reason="manual profile matched device capabilities",
    )
    state = CaptureRuntimeState(
        available=True,
        device="/dev/video0",
        profile=profile,
        backend="gst-resource:nvmm-mjpg-iomode2",
    )
    app.state.runtime.running = True
    app.state.runtime.pipeline = SimpleNamespace(
        running=True,
        stop=lambda: None,
    )
    app.state.capture = SimpleNamespace(
        config=cfg.capture,
        roi_size=cfg.roi.size,
        source=object(),
        session=SimpleNamespace(running=True),
        state=state,
        last_config_error=None,
        configure_profile_only=lambda *args, **kwargs: state,
    )

    def fail_create(*args, **kwargs):
        raise RuntimeError("native bridge unavailable")

    monkeypatch.setattr(
        "novasight.runtime.reconfigurator.create_runtime_pipeline",
        fail_create,
    )

    report = RuntimeReconfigurator(app).select_capture(
        device="/dev/video0",
        preference="manual",
        pixel_format="MJPG",
        width=2560,
        height=1440,
        fps=120,
    )

    assert report.applied is False
    assert report.rolled_back is False
    assert report.message.startswith("采集配置已应用，但启动主链失败")
    assert any(
        section.section == "runtime_pipeline" and section.status == "failed"
        for section in report.sections
    )
    assert app.state.runtime.running is False


def test_inference_runtime_injects_gpu_preprocessor_into_current_engine() -> None:
    engine = _FakeEngine()
    preprocessor = JetsonGpuResourcePreprocessor()

    runtime = InferenceRuntime(engine, gpu_preprocessor=preprocessor)

    assert engine.gpu_preprocessor is preprocessor
    assert runtime.status()["gpu_preprocessor"]["selected"] == "jetson_nvmm_cuda"


def test_inference_runtime_preserves_preprocessor_on_threshold_reconfigure() -> None:
    engine = _FakeEngine()
    preprocessor = JetsonGpuResourcePreprocessor()
    runtime = InferenceRuntime(engine, gpu_preprocessor=preprocessor)
    engine.gpu_preprocessor = None

    runtime.configure(confidence_threshold=0.5, nms_threshold=0.4)

    assert engine.confidence_threshold == 0.5
    assert engine.nms_threshold == 0.4
    assert engine.gpu_preprocessor is preprocessor


def test_inference_runtime_passes_preprocessor_to_tensorrt_candidates() -> None:
    runtime = InferenceRuntime(_FakeEngine())
    preprocessor = JetsonGpuResourcePreprocessor()

    runtime.configure(gpu_preprocessor=preprocessor)
    candidate = runtime._engine_for_artifact(Path("model.engine"))

    assert isinstance(candidate, TensorRtInferenceEngine)
    assert candidate._gpu_preprocessor is preprocessor


def test_nvmm_gstreamer_sample_uses_configured_gpu_preprocessor(
    monkeypatch,
) -> None:
    class Owner:
        def release(self) -> None:
            pass

    calls: list[dict] = []

    def prepare_native_tensor(payload: dict) -> dict:
        calls.append(dict(payload))
        return {
            "device_ptr": 12345,
            "nbytes": 1 * 3 * 320 * 320 * 4,
            "shape": (1, 3, 320, 320),
            "dtype": "float32",
            "zero_copy": True,
            "memory_space": "cuda_device",
            "backend": "jetson_cuda",
            "owner": Owner(),
        }

    module_name = "configured_gpu_preprocessor_bridge_for_test"
    monkeypatch.setitem(
        sys.modules,
        module_name,
        SimpleNamespace(
            ABI_VERSION=1,
            CAPABILITIES={
                "memory": ["nvmm", "dmabuf"],
                "resource_kind": ["gstreamer_sample"],
                "resource_source": ["appsink"],
                "formats": ["NV12"],
                "dtype": ["float32"],
            },
            status=lambda: {
                "available": True,
                "ready": True,
                "backend": "jetson_cuda",
                "zero_copy": True,
                "memory_space": "cuda_device",
            },
            prepare_tensor=prepare_native_tensor,
        ),
    )
    monkeypatch.setenv("NOVASIGHT_JETSON_PREPROCESSOR", module_name)
    cfg = RuntimeConfig()
    cfg.inference.backend = "nvmm_latest"
    cfg.capture.memory = "nvmm"
    preprocessor = create_gpu_resource_preprocessor(cfg)
    shape = TensorInputShape(batch=1, channels=3, height=320, width=320)
    prepared = PreparedTensorInput(
        mode="gpu_buffer",
        buffer=object(),
        frame_id=10,
        capture_ts_ns=1_000_000_000,
        width=320,
        height=320,
        pixel_format="NV12",
        source_width=1920,
        source_height=1080,
        offset_x=800,
        offset_y=300,
        needs_resize=False,
        resource_kind="gstreamer_sample",
        resource_memory="nvmm",
        resource_source="appsink",
        dmabuf_fd=7,
        resource_metadata={},
        resource_width=320,
        resource_height=320,
        resource_pixel_format="NV12",
    )

    result = prepare_tensor(prepared, shape, gpu_preprocessor=preprocessor)

    assert result.backend == "jetson_cuda"
    assert result.location == "device"
    assert result.zero_copy is True
    assert result.tensor.device_ptr == 12345
    assert calls[0]["resource_kind"] == "gstreamer_sample"
    assert calls[0]["resource_memory"] == "nvmm"


def test_nvmm_frame_resource_keeps_gstreamer_sample_handle_alive() -> None:
    sample = object()
    resource = FrameResource(
        kind="gstreamer_sample",
        handle=sample,
        memory="nvmm",
        width=640,
        height=640,
        pixel_format="NV12",
        source="appsink",
        dmabuf_fd=7,
    )
    frame = CapturedFrame(
        frame_id=10,
        width=640,
        height=640,
        pixel_format="NV12",
        ts_ns=1_000_000_000,
        capture_wait_ms=0.1,
        image=None,
        frame_resource=resource,
    )
    prepared = prepare_tensor_input(
        frame,
        TensorInputShape(batch=1, channels=3, height=320, width=320),
    )

    assert frame.image is None
    assert frame.frame_resource.handle is sample
    assert frame.gpu_buffer is sample
    assert prepared.buffer is sample
    assert prepared.dmabuf_fd == 7


def test_cpu_compatible_frame_resource_owns_host_copy_not_gstreamer_sample() -> None:
    import numpy as np

    image = np.zeros((2, 2, 3), dtype=np.uint8)
    resource = _host_frame_resource_from_copy(
        image,
        width=2,
        height=2,
        pixel_format="BGR",
        source_pixel_format="BGRx",
        caps_string="video/x-raw,format=BGRx,width=2,height=2",
        copy_cost_ms=0.25,
        pipeline_string="v4l2src ! appsink",
    )
    frame = CapturedFrame(
        frame_id=10,
        width=2,
        height=2,
        pixel_format="BGR",
        ts_ns=1_000_000_000,
        capture_wait_ms=0.1,
        image=image,
        frame_resource=resource,
    )

    prepared = prepare_tensor_input(
        frame,
        TensorInputShape(batch=1, channels=3, height=2, width=2),
    )

    assert resource.kind == "host_frame"
    assert resource.memory == "cpu"
    assert resource.handle is image
    assert resource.metadata["source_pixel_format"] == "BGRX"
    assert resource.metadata["copy_cost_ms"] == pytest.approx(0.25)
    assert prepared.mode == "host_frame"
    assert prepared.buffer is image
    assert prepared.resource_kind == "host_frame"
    assert prepared.resource_memory == "cpu"
    assert prepared.resource_source == "appsink"
    assert prepared.gst_buffer_ptr is None


def test_cpu_host_frame_preprocess_converts_bgrx_to_rgb_nchw_fp16() -> None:
    import numpy as np

    image = np.array(
        [
            [[10, 20, 30, 255], [1, 2, 3, 255]],
            [[90, 80, 70, 255], [7, 8, 9, 255]],
        ],
        dtype=np.uint8,
    )
    frame = CapturedFrame(
        frame_id=33,
        width=2,
        height=2,
        pixel_format="BGRx",
        ts_ns=1_000_000_000,
        capture_wait_ms=0.1,
        image=image,
    )
    shape = TensorInputShape(batch=1, channels=3, height=2, width=2, dtype="float16")

    prepared = prepare_tensor_input(frame, shape)
    result = prepare_host_tensor(prepared, shape)

    assert prepared.mode == "host_frame"
    assert prepared.resource_memory == "cpu"
    assert result.backend == "cpu"
    assert result.location == "host"
    assert result.zero_copy is False
    assert result.tensor.shape == (1, 3, 2, 2)
    assert result.tensor.dtype == np.float16
    assert result.tensor[0, 0, 0, 0] == np.float16(30 / 255.0)
    assert result.tensor[0, 1, 0, 0] == np.float16(20 / 255.0)
    assert result.tensor[0, 2, 0, 0] == np.float16(10 / 255.0)
    assert result.timings is not None
    assert result.timings["total_ms"] >= 0.0


def test_cpu_host_frame_preprocess_letterboxes_non_square_input() -> None:
    import numpy as np

    image = np.zeros((2, 4, 3), dtype=np.uint8)
    image[:, :, 2] = 255
    frame = CapturedFrame(
        frame_id=34,
        width=4,
        height=2,
        pixel_format="BGR",
        ts_ns=1_000_000_000,
        capture_wait_ms=0.1,
        image=image,
    )
    shape = TensorInputShape(batch=1, channels=3, height=4, width=4, dtype="float32")

    prepared = prepare_tensor_input(frame, shape)
    result = prepare_host_tensor(prepared, shape)

    assert result.metadata is not None
    assert result.metadata["resize_mode"] == "letterbox"
    assert result.metadata["model_content_width"] == 4
    assert result.metadata["model_content_height"] == 2
    assert result.metadata["pad_x"] == 0
    assert result.metadata["pad_y"] == 1
    assert result.tensor.shape == (1, 3, 4, 4)
    assert result.tensor[0, 0, 0, 0] == pytest.approx(114 / 255.0)
    assert result.tensor[0, 0, 1, 0] == pytest.approx(1.0)


def test_model_to_roi_mapping_accounts_for_letterbox_padding() -> None:
    prepared = PreparedTensorInput(
        mode="host_frame",
        buffer=object(),
        frame_id=34,
        capture_ts_ns=1_000_000_000,
        width=400,
        height=200,
        pixel_format="BGR",
        source_width=1920,
        source_height=1080,
        offset_x=760,
        offset_y=440,
        needs_resize=True,
    )
    preprocess_result = TensorPreprocessResult(
        tensor=object(),
        backend="cpu",
        input_mode="host_frame",
        resource_kind="host_frame",
        resource_memory="cpu",
        metadata={
            "resize_mode": "letterbox",
            "model_content_width": 400,
            "model_content_height": 200,
            "pad_x": 0,
            "pad_y": 100,
        },
    )
    detections = [
        InferenceDetection(
            cls=0,
            score=0.9,
            box=BBox.from_xyxy(100, 150, 300, 250),
        )
    ]

    mapped = map_model_detections_to_roi_frame(
        detections,
        prepared=prepared,
        shape=TensorInputShape(batch=1, channels=3, height=400, width=400),
        preprocess_result=preprocess_result,
    )

    assert mapped[0].box.x1 == pytest.approx(100.0)
    assert mapped[0].box.y1 == pytest.approx(50.0)
    assert mapped[0].box.x2 == pytest.approx(300.0)
    assert mapped[0].box.y2 == pytest.approx(150.0)


def test_gstreamer_cpu_sample_copy_accepts_rgba_rgb_and_i420_fallback_formats() -> None:
    import numpy as np

    class Gst:
        class MapFlags:
            READ = object()

    class FakeStructure:
        def __init__(self, fmt: str, width: int, height: int) -> None:
            self.fmt = fmt
            self.width = width
            self.height = height

        def get_int(self, key: str):
            return True, self.width if key == "width" else self.height

        def get_string(self, key: str) -> str:
            assert key == "format"
            return self.fmt

    class FakeCaps:
        def __init__(self, fmt: str, width: int, height: int) -> None:
            self.structure = FakeStructure(fmt, width, height)

        def get_size(self) -> int:
            return 1

        def get_structure(self, _index: int) -> FakeStructure:
            return self.structure

        def get_features(self, _index: int):
            return SimpleNamespace(to_string=lambda: "")

    class FakeBuffer:
        def __init__(self, payload: bytes) -> None:
            self.payload = payload

        def n_memory(self) -> int:
            return 0

        def map(self, _flags):
            return True, SimpleNamespace(data=self.payload)

        def unmap(self, _info) -> None:
            pass

    class FakeSample:
        def __init__(self, fmt: str, width: int, height: int, payload: bytes) -> None:
            self.caps = FakeCaps(fmt, width, height)
            self.buffer = FakeBuffer(payload)

        def get_caps(self) -> FakeCaps:
            return self.caps

        def get_buffer(self) -> FakeBuffer:
            return self.buffer

    rgba = np.array([[[1, 2, 3, 255]]], dtype=np.uint8)
    rgb = np.array([[[4, 5, 6]]], dtype=np.uint8)
    y = np.array([[82, 82], [82, 82]], dtype=np.uint8)
    u = np.array([[90]], dtype=np.uint8)
    v = np.array([[240]], dtype=np.uint8)
    i420 = bytes([*y.reshape(-1), *u.reshape(-1), *v.reshape(-1)])

    rgba_image, _, _, rgba_fmt = _sample_to_bgr(FakeSample("RGBA", 1, 1, rgba.tobytes()), Gst)
    rgb_image, _, _, rgb_fmt = _sample_to_bgr(FakeSample("RGB", 1, 1, rgb.tobytes()), Gst)
    i420_image, _, _, i420_fmt = _sample_to_bgr(FakeSample("I420", 2, 2, i420), Gst)

    assert rgba_fmt == "RGBA"
    assert rgba_image.tolist() == [[[3, 2, 1]]]
    assert rgb_fmt == "RGB"
    assert rgb_image.tolist() == [[[6, 5, 4]]]
    assert i420_fmt == "I420"
    assert i420_image.shape == (2, 2, 3)
    assert i420_image[0, 0, 2] > 200


def test_gstreamer_cpu_sample_copy_owns_bgr_memory_after_unmap() -> None:
    class Gst:
        class MapFlags:
            READ = object()

    class FakeStructure:
        def get_int(self, key: str):
            return True, 1

        def get_string(self, key: str) -> str:
            assert key == "format"
            return "BGR"

    class FakeCaps:
        def get_size(self) -> int:
            return 1

        def get_structure(self, _index: int) -> FakeStructure:
            return FakeStructure()

        def get_features(self, _index: int):
            return SimpleNamespace(to_string=lambda: "")

    class FakeBuffer:
        def __init__(self, payload: bytearray) -> None:
            self.payload = payload
            self.unmapped = False

        def n_memory(self) -> int:
            return 0

        def map(self, _flags):
            return True, SimpleNamespace(data=self.payload)

        def unmap(self, _info) -> None:
            self.unmapped = True

    class FakeSample:
        def __init__(self, payload: bytearray) -> None:
            self.buffer = FakeBuffer(payload)

        def get_caps(self) -> FakeCaps:
            return FakeCaps()

        def get_buffer(self) -> FakeBuffer:
            return self.buffer

    payload = bytearray([10, 20, 30])
    sample = FakeSample(payload)

    image, _, _, fmt = _sample_to_bgr(sample, Gst)
    payload[:] = b"\x01\x02\x03"

    assert fmt == "BGR"
    assert sample.buffer.unmapped is True
    assert image.tolist() == [[[10, 20, 30]]]


def test_nvmm_frame_resource_records_gst_buffer_pointer_when_dmabuf_is_absent() -> None:
    class FakeFeatures:
        def to_string(self) -> str:
            return "memory:NVMM"

    class FakeCaps:
        def get_features(self, _index: int) -> FakeFeatures:
            return FakeFeatures()

    class FakeBuffer:
        pts = 11
        dts = 22
        # Simulate pygobject exposing a real GObject pointer via __gpointer__.
        # The capture pipeline relies on this attribute rather than hash(obj)
        # — the latter is Python's object id, not a real GstBuffer*, and
        # would make the native bridge reinterpret_cast a garbage value
        # and segfault inside gst_buffer_ref.
        __gpointer__ = 0x7F00_AB12_3456_0000

        def n_memory(self) -> int:
            return 0

    class FakeSample:
        def __init__(self) -> None:
            self.buffer = FakeBuffer()

        def get_buffer(self) -> FakeBuffer:
            return self.buffer

        def get_caps(self) -> FakeCaps:
            return FakeCaps()

    resource = _sample_frame_resource(
        FakeSample(),
        SimpleNamespace(),
        width=640,
        height=640,
        pixel_format="NV12",
    )

    assert resource is not None
    assert resource.memory == "nvmm"
    assert resource.dmabuf_fd is None
    assert resource.gst_buffer_ptr == 0x7F00_AB12_3456_0000
    assert resource.metadata["gst_buffer_ptr"] == 0x7F00_AB12_3456_0000


def test_nvmm_frame_resource_omits_gst_buffer_pointer_without_real_pointer_attr() -> None:
    # When the buffer exposes neither __gpointer__ nor __pointer__ nor gpointer,
    # capture MUST NOT synthesize a pseudo-pointer from hash(obj). Dropping the
    # hash() fallback means gst_buffer_ptr is None, which forces the bridge to
    # raise a clean error instead of segfaulting on a bad cast.
    class FakeFeatures:
        def to_string(self) -> str:
            return "memory:NVMM"

    class FakeCaps:
        def get_features(self, _index: int) -> FakeFeatures:
            return FakeFeatures()

    class FakeBuffer:
        def __hash__(self) -> int:
            return 987654

        def n_memory(self) -> int:
            return 0

    class FakeSample:
        def __init__(self) -> None:
            self.buffer = FakeBuffer()

        def get_buffer(self) -> FakeBuffer:
            return self.buffer

        def get_caps(self) -> FakeCaps:
            return FakeCaps()

    resource = _sample_frame_resource(
        FakeSample(),
        SimpleNamespace(),
        width=640,
        height=640,
        pixel_format="NV12",
    )

    assert resource is not None
    assert resource.memory == "nvmm"
    assert resource.dmabuf_fd is None
    assert resource.gst_buffer_ptr is None
    assert resource.metadata["gst_buffer_ptr"] is None


def test_nvmm_gstreamer_sample_without_dmabuf_uses_gst_buffer_pointer_fallback(
    monkeypatch,
) -> None:
    class Owner:
        def release(self) -> None:
            pass

    calls: list[dict] = []

    def prepare_native_tensor(payload: dict) -> dict:
        calls.append(dict(payload))
        return {
            "device_ptr": 12345,
            "nbytes": 1 * 3 * 320 * 320 * 4,
            "shape": (1, 3, 320, 320),
            "dtype": "float32",
            "zero_copy": True,
            "memory_space": "cuda_device",
            "backend": "jetson_cuda",
            "owner": Owner(),
        }

    module_name = "gst_buffer_ptr_gpu_preprocessor_bridge_for_test"
    monkeypatch.setitem(
        sys.modules,
        module_name,
        SimpleNamespace(
            ABI_VERSION=1,
            CAPABILITIES={
                "memory": ["nvmm"],
                "resource_kind": ["gstreamer_sample"],
                "resource_source": ["appsink"],
                "formats": ["NV12"],
                "dtype": ["float32"],
            },
            status=lambda: {
                "available": True,
                "ready": True,
                "backend": "jetson_cuda",
                "zero_copy": True,
                "memory_space": "cuda_device",
            },
            prepare_tensor=prepare_native_tensor,
        ),
    )
    monkeypatch.setenv("NOVASIGHT_JETSON_PREPROCESSOR", module_name)
    cfg = RuntimeConfig()
    cfg.inference.backend = "nvmm_latest"
    cfg.capture.memory = "nvmm"
    preprocessor = create_gpu_resource_preprocessor(cfg)
    shape = TensorInputShape(batch=1, channels=3, height=320, width=320)
    prepared = PreparedTensorInput(
        mode="gpu_buffer",
        buffer=object(),
        frame_id=10,
        capture_ts_ns=1_000_000_000,
        width=640,
        height=640,
        pixel_format="NV12",
        source_width=2560,
        source_height=1440,
        offset_x=960,
        offset_y=400,
        needs_resize=True,
        resource_kind="gstreamer_sample",
        resource_memory="nvmm",
        resource_source="appsink",
        dmabuf_fd=None,
        resource_metadata={"gst_buffer_ptr": 987654},
        resource_width=640,
        resource_height=640,
        resource_pixel_format="NV12",
    )

    result = prepare_tensor(prepared, shape, gpu_preprocessor=preprocessor)

    assert result.zero_copy is True
    assert calls[0]["dmabuf_fd"] is None
    assert calls[0]["gst_buffer_ptr"] == 987654


def test_nvmm_gstreamer_sample_requires_live_resource_handle(
    monkeypatch,
) -> None:
    calls: list[dict] = []

    module_name = "resource_handle_required_gpu_preprocessor_bridge_for_test"
    monkeypatch.setitem(
        sys.modules,
        module_name,
        SimpleNamespace(
            ABI_VERSION=1,
            CAPABILITIES={
                "memory": ["nvmm"],
                "resource_kind": ["gstreamer_sample"],
                "resource_source": ["appsink"],
                "formats": ["NV12"],
                "dtype": ["float32"],
            },
            status=lambda: {
                "available": True,
                "ready": True,
                "backend": "jetson_cuda",
                "zero_copy": True,
                "memory_space": "cuda_device",
            },
            prepare_tensor=lambda payload: calls.append(dict(payload)),
        ),
    )
    monkeypatch.setenv("NOVASIGHT_JETSON_PREPROCESSOR", module_name)
    prepared = PreparedTensorInput(
        mode="gpu_buffer",
        buffer=None,
        frame_id=10,
        capture_ts_ns=1_000_000_000,
        width=320,
        height=320,
        pixel_format="NV12",
        source_width=1920,
        source_height=1080,
        offset_x=800,
        offset_y=300,
        needs_resize=False,
        resource_kind="gstreamer_sample",
        resource_memory="nvmm",
        resource_source="appsink",
        dmabuf_fd=7,
        resource_metadata={},
        resource_width=320,
        resource_height=320,
        resource_pixel_format="NV12",
    )
    cfg = RuntimeConfig()
    cfg.inference.backend = "nvmm_latest"
    cfg.capture.memory = "nvmm"
    preprocessor = create_gpu_resource_preprocessor(cfg)
    shape = TensorInputShape(batch=1, channels=3, height=320, width=320)

    with pytest.raises(Exception, match="resource_handle"):
        prepare_tensor(prepared, shape, gpu_preprocessor=preprocessor)

    assert calls == []


def test_nvmm_native_preprocess_timings_surface_in_result_debug(
    monkeypatch,
) -> None:
    class Owner:
        def release(self) -> None:
            pass

    def prepare_native_tensor(payload: dict) -> dict:
        return {
            "device_ptr": 12345,
            "nbytes": 1 * 3 * 320 * 320 * 4,
            "shape": (1, 3, 320, 320),
            "dtype": "float32",
            "zero_copy": True,
            "memory_space": "cuda_device",
            "backend": "jetson_cuda",
            "owner": Owner(),
            "timings": {
                "nvbufsurftransform_ms": 0.42,
                "cuda_kernel_ms": 0.13,
                "total_ms": 0.71,
            },
        }

    module_name = "timed_gpu_preprocessor_bridge_for_test"
    monkeypatch.setitem(
        sys.modules,
        module_name,
        SimpleNamespace(
            ABI_VERSION=1,
            CAPABILITIES={
                "memory": ["nvmm"],
                "resource_kind": ["gstreamer_sample"],
                "resource_source": ["appsink"],
                "formats": ["NV12"],
                "dtype": ["float32"],
            },
            status=lambda: {
                "available": True,
                "ready": True,
                "backend": "jetson_cuda",
                "zero_copy": True,
                "memory_space": "cuda_device",
            },
            prepare_tensor=prepare_native_tensor,
        ),
    )
    monkeypatch.setenv("NOVASIGHT_JETSON_PREPROCESSOR", module_name)
    cfg = RuntimeConfig()
    cfg.inference.backend = "nvmm_latest"
    cfg.capture.memory = "nvmm"
    preprocessor = create_gpu_resource_preprocessor(cfg)
    shape = TensorInputShape(batch=1, channels=3, height=320, width=320)
    prepared = PreparedTensorInput(
        mode="gpu_buffer",
        buffer=object(),
        frame_id=10,
        capture_ts_ns=1_000_000_000,
        width=320,
        height=320,
        pixel_format="NV12",
        source_width=1920,
        source_height=1080,
        offset_x=800,
        offset_y=300,
        needs_resize=False,
        resource_kind="gstreamer_sample",
        resource_memory="nvmm",
        resource_source="appsink",
        dmabuf_fd=7,
        resource_metadata={},
        resource_width=320,
        resource_height=320,
        resource_pixel_format="NV12",
    )

    result = prepare_tensor(prepared, shape, gpu_preprocessor=preprocessor)

    debug = result.debug_payload()
    assert debug["preprocess_native_timings"]["nvbufsurftransform_ms"] == 0.42
    assert debug["preprocess_native_timings"]["cuda_kernel_ms"] == 0.13
    assert debug["preprocess_native_timings"]["total_ms"] == 0.71


def test_tensorrt_inference_debug_includes_native_preprocess_timings(
    monkeypatch,
) -> None:
    prepared = PreparedTensorInput(
        mode="gpu_buffer",
        buffer=object(),
        frame_id=10,
        capture_ts_ns=1_000_000_000,
        width=320,
        height=320,
        pixel_format="NV12",
        source_width=1920,
        source_height=1080,
        offset_x=800,
        offset_y=300,
        needs_resize=False,
        resource_kind="gstreamer_sample",
        resource_memory="nvmm",
        resource_source="appsink",
        dmabuf_fd=7,
        resource_metadata={},
        resource_width=320,
        resource_height=320,
        resource_pixel_format="NV12",
    )
    shape = TensorInputShape(batch=1, channels=3, height=320, width=320)
    preprocess_result = TensorPreprocessResult(
        tensor=DeviceTensor(
            device_ptr=12345,
            nbytes=shape.nbytes,
            shape=(1, 3, 320, 320),
            owner=SimpleNamespace(release=lambda: None),
        ),
        backend="jetson_cuda",
        input_mode="gpu_buffer",
        resource_kind="gstreamer_sample",
        resource_memory="nvmm",
        location="device",
        zero_copy=True,
        timings={
            "nvbufsurftransform_ms": 0.42,
            "cuda_kernel_ms": 0.13,
            "total_ms": 0.71,
        },
    )
    engine = TensorRtInferenceEngine()
    engine._loaded = True
    engine._input_shape = shape
    engine._context = object()
    engine._cudart = object()
    engine._classes = ["target"]
    engine._input_name = "images"
    engine._output_name = "output0"
    engine._output_shape = (1, 6, 0)
    engine._output_dtype = "float32"

    monkeypatch.setattr(tensorrt_module, "prepare_tensor_input", lambda _frame, _shape: prepared)
    monkeypatch.setattr(
        tensorrt_module,
        "prepare_tensor",
        lambda _prepared, _shape, *, gpu_preprocessor: preprocess_result,
    )
    monkeypatch.setattr(
        engine, "_execute", lambda _tensor: ([], {"timings": {"input_location": "device"}})
    )

    result = engine.infer(object())

    assert result.available is True
    assert result.debug["preprocess"]["preprocess_native_timings"]["cuda_kernel_ms"] == 0.13
    assert result.debug["timings"]["native_preprocess_nvbufsurftransform_ms"] == 0.42
    assert result.debug["timings"]["native_preprocess_cuda_kernel_ms"] == 0.13
    assert result.debug["timings"]["native_preprocess_total_ms"] == 0.71
    assert engine.status()["last_preprocess_timings"] == {
        "nvbufsurftransform_ms": 0.42,
        "cuda_kernel_ms": 0.13,
        "total_ms": 0.71,
    }


def test_native_ctypes_bridge_auto_discovers_default_build_output(
    tmp_path,
    monkeypatch,
) -> None:
    import novasight_jetson_preprocess_native as native_backend

    library = tmp_path / "build" / "jetson-native" / "libnovasight_preprocess.so"
    library.parent.mkdir(parents=True)
    library.write_text("not a shared object", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(native_backend.LIBRARY_ENV, raising=False)
    native_backend._reset_library_cache()

    try:
        status = native_backend.status()
    finally:
        native_backend._reset_library_cache()

    assert status["available"] is False
    assert status["reason"] == "native_library_unavailable"
    assert status["library"] == str(library)


def test_native_ctypes_bridge_auto_builds_default_library_on_jetson(
    tmp_path,
    monkeypatch,
) -> None:
    import novasight_jetson_preprocess_native as native_backend

    library = tmp_path / "build" / "jetson-native" / "libnovasight_preprocess.so"
    calls: list[list[str]] = []

    def fake_run(command: list[str], timeout_s: float):
        calls.append(list(command))
        if "--build" in command:
            library.parent.mkdir(parents=True, exist_ok=True)
            library.write_text("not a shared object", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(native_backend.LIBRARY_ENV, raising=False)
    monkeypatch.setattr(native_backend, "_is_jetson_runtime", lambda: True)
    monkeypatch.setattr(native_backend, "_run_auto_build_command", fake_run)
    native_backend._reset_library_cache()

    try:
        status = native_backend.status()
    finally:
        native_backend._reset_library_cache()

    assert len(calls) == 2
    assert calls[0][:4] == [
        "cmake",
        "-S",
        str(Path(native_backend.__file__).resolve().parent / "native"),
        "-B",
    ]
    assert calls[0][4] == str(tmp_path / "build" / "jetson-native")
    assert calls[1] == ["cmake", "--build", str(tmp_path / "build" / "jetson-native")]
    assert status["available"] is False
    assert status["reason"] == "native_library_unavailable"
    assert status["library"] == str(library)


def test_native_ctypes_bridge_does_not_auto_build_off_jetson(
    tmp_path,
    monkeypatch,
) -> None:
    import novasight_jetson_preprocess_native as native_backend

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(native_backend.LIBRARY_ENV, raising=False)
    monkeypatch.setattr(native_backend, "_is_jetson_runtime", lambda: False)
    monkeypatch.setattr(
        native_backend,
        "_run_auto_build_command",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not build off Jetson")),
    )
    native_backend._reset_library_cache()

    try:
        status = native_backend.status()
    finally:
        native_backend._reset_library_cache()

    assert status["available"] is False
    assert status["reason"] == "native_implementation_missing"
    assert status["library"] == ""
