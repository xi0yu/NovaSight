from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from novasight.api import create_app
from novasight.config import RuntimeConfig
from novasight.inference.input import PreparedTensorInput, TensorInputShape
from novasight.inference.jetson import (
    JetsonGpuResourcePreprocessor,
    create_gpu_resource_preprocessor,
)
from novasight.inference.preprocess import prepare_tensor
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


def test_native_ctypes_bridge_auto_discovers_default_build_output(
    tmp_path,
    monkeypatch,
) -> None:
    import novasight_jetson_preprocess_native as native_backend

    library = tmp_path / "build" / "jetson-native" / "libnovasight_jetson_preprocess_native.so"
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
