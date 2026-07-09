from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from novasight.api import create_app
from novasight.capture.source import CapturedFrame, FrameResource, _sample_frame_resource
from novasight.capture.state import CaptureProfile, CaptureRuntimeState
from novasight.config import RuntimeConfig
from novasight.inference.input import PreparedTensorInput, TensorInputShape, prepare_tensor_input
from novasight.inference.jetson import (
    JetsonGpuResourcePreprocessor,
    create_gpu_resource_preprocessor,
)
from novasight.inference.preprocess import DeviceTensor, TensorPreprocessResult, prepare_tensor
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
        roi_offset_x=cfg.roi.offset_x,
        roi_offset_y=cfg.roi.offset_y,
        source=object(),
        session=SimpleNamespace(running=True),
        state=state,
        last_config_error=None,
        configure=lambda *args, **kwargs: state,
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
    cfg.inference.enabled = True
    cfg.inference.backend = "nvmm_latest"
    cfg.capture.memory = "nvmm"
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
        roi_offset_x=cfg.roi.offset_x,
        roi_offset_y=cfg.roi.offset_y,
        source=object(),
        session=SimpleNamespace(running=True),
        state=state,
        last_config_error=None,
        configure=lambda *args, **kwargs: state,
    )

    def fail_start(*args, **kwargs):
        raise RuntimeError("native bridge unavailable")

    monkeypatch.setattr("novasight.runtime.pipeline.RuntimePipeline.start", fail_start)

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
    assert any(section.section == "runtime_pipeline" and section.status == "failed" for section in report.sections)
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
    assert resource.gst_buffer_ptr == 987654
    assert resource.metadata["gst_buffer_ptr"] == 987654


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
    monkeypatch.setattr(engine, "_execute", lambda _tensor: ([], {"timings": {"input_location": "device"}}))

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
    assert calls[0][:4] == ["cmake", "-S", str(Path(native_backend.__file__).resolve().parent / "native"), "-B"]
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
