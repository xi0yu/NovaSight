"""Tests for runtime configuration loading and validation.

Covers the schema-enforced contract: round-trip, missing/empty inputs,
unknown keys, and type errors. Defaults are exercised in one test, and
unknown-key rejection is parametrized.
"""
from pathlib import Path

import pytest

from novasight.config import (
    RuntimeConfig,
    load_runtime_config,
    parse_runtime_config,
    save_runtime_config,
)
from novasight.config.schema import runtime_config_schema
from novasight.capture.pipeline import (
    LATEST_ONLY_QUEUE,
    build_appsink_candidates,
    build_pipeline_candidates,
    build_resource_appsink_candidates,
)
from novasight.capture.state import CaptureProfile


def test_packaging_includes_jetson_bridge_adapter_packages() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert 'include = ["novasight*", "novasight_jetson_preprocess*"]' in pyproject
    assert (
        'novasight_jetson_preprocess_native = ["include/*.h", '
        '"native/CMakeLists.txt", "native/src/*.cpp", '
        '"native/src/jetson/*.cpp", "native/src/jetson/*.cu", '
        '"native/src/jetson/*.h", "native/README.md", "native/IMPLEMENTATION.md"]'
        in pyproject
    )
    assert Path(
        "novasight_jetson_preprocess_native/include/"
        "novasight_jetson_preprocess_native.h"
    ).is_file()
    header = Path(
        "novasight_jetson_preprocess_native/include/"
        "novasight_jetson_preprocess_native.h"
    ).read_text(encoding="utf-8")
    assert "NOVASIGHT_JETSON_PREPROCESS_ABI_VERSION 1" in header
    assert "uint32_t novasight_abi_version(void)" in header
    assert Path("novasight_jetson_preprocess_native/native/CMakeLists.txt").is_file()
    assert Path(
        "novasight_jetson_preprocess_native/native/src/"
        "novasight_jetson_preprocess_native.cpp"
    ).is_file()
    assert Path(
        "novasight_jetson_preprocess_native/native/src/jetson/"
        "novasight_jetson_preprocess_native_jetson_scaffold.cpp"
    ).is_file()
    assert Path(
        "novasight_jetson_preprocess_native/native/src/jetson/"
        "novasight_jetson_preprocess_native_jetson_support.h"
    ).is_file()
    assert Path(
        "novasight_jetson_preprocess_native/native/src/jetson/"
        "novasight_jetson_preprocess_native_jetson_support.cpp"
    ).is_file()
    assert Path(
        "novasight_jetson_preprocess_native/native/src/jetson/"
        "novasight_jetson_preprocess_native_jetson_cuda.cu"
    ).is_file()
    assert Path("novasight_jetson_preprocess_native/native/IMPLEMENTATION.md").is_file()


def test_native_jetson_preprocess_cmake_has_production_gate() -> None:
    cmake = Path(
        "novasight_jetson_preprocess_native/native/CMakeLists.txt"
    ).read_text(encoding="utf-8")

    assert 'NOVASIGHT_JETSON_PREPROCESS_IMPL "reference"' in cmake
    assert "reference jetson_scaffold jetson" in cmake
    assert "NOVASIGHT_JETSON_PREPROCESS_PRODUCTION_SOURCE" in cmake
    assert "CUDAToolkit" in cmake
    assert "enable_language(CUDA)" in cmake
    assert "CMAKE_CUDA_STANDARD 17" in cmake
    assert "cuda_std_17" in cmake
    assert "CUDA::cuda_driver" in cmake
    assert "PkgConfig" in cmake
    assert "gstreamer-1.0" in cmake
    assert "nvbufsurface.h" in cmake
    assert "nvbufsurftransform" in cmake
    assert "NOVASIGHT_JETSON_PREPROCESS_SUPPORT_SOURCES" in cmake
    assert "novasight_jetson_preprocess_native_jetson_support.cpp" in cmake
    assert "${CMAKE_CURRENT_LIST_DIR}/src/jetson" in cmake
    assert "OUTPUT_NAME novasight_preprocess" in cmake
    assert "jetson_cuda_preprocess_not_compiled" not in cmake


def test_native_jetson_cuda_source_uses_real_dmabuf_egl_cuda_path() -> None:
    cu_path = Path(
        "novasight_jetson_preprocess_native/native/src/jetson/"
        "novasight_jetson_preprocess_native_jetson_cuda.cu"
    )
    cpp_path = Path(
        "novasight_jetson_preprocess_native/native/src/jetson/"
        "novasight_jetson_preprocess_native_jetson_gst_buffer.cpp"
    )
    cu_source = cu_path.read_text(encoding="utf-8")
    cpp_source = cpp_path.read_text(encoding="utf-8")

    assert "NvBufSurfaceFromFd" in cu_source
    assert "nvbufsurftransform.h" in cu_source.lower()
    # The .cu must NOT pull <gst/gst.h> directly: nvcc 12.6 fails to parse
    # glib-2.0/gmacros.h's legacy __has_attribute() macro. All GStreamer
    # access must live in the sibling .cpp compiled with g++.
    assert "#include <gst/gst.h>" not in cu_source
    assert "gst_buffer_ref(" not in cu_source
    assert "gst_buffer_unref(" not in cu_source
    assert "gst_buffer_map(" not in cu_source
    assert "gst_buffer_unmap(" not in cu_source
    assert cpp_path.exists(), "gst_buffer helper .cpp is missing"
    assert "#include <gst/gst.h>" in cpp_source
    assert "gst_buffer_map" in cpp_source
    assert "gst_buffer_ref" in cpp_source
    assert "NvBufSurfTransform" in cu_source
    assert "NVBUF_COLOR_FORMAT_RGBA" in cu_source
    assert "rgba_to_nchw_kernel" in cu_source
    assert '\\"timings\\"' in cu_source
    assert '\\"nvbufsurftransform_ms\\"' in cu_source
    assert '\\"egl_cuda_map_ms\\"' in cu_source
    assert '\\"cuda_kernel_ms\\"' in cu_source
    assert "NvBufSurfaceMapEglImage" in cu_source
    assert "cuGraphicsEGLRegisterImage" in cu_source
    assert "cuGraphicsResourceGetMappedEglFrame" in cu_source
    assert "cudaMemcpy2DFromArrayAsync" in cu_source
    assert "surface->surfaceList[0].width" in cu_source
    assert "surface->surfaceList[0].height" in cu_source
    assert "nvbufsurface_geometry_mismatch" in cu_source
    assert "bool validate_rgba_plane_layout" in cu_source
    assert "cuda_egl_rgba_plane_invalid" in cu_source
    assert "rgba_pitch < width * 4" in cu_source
    assert "__device__ int scaled_source_index" in cu_source
    assert "nv12_to_nchw_kernel" not in cu_source
    assert "max(0, min(255" not in cu_source
    assert "const int src_x = min(" not in cu_source
    assert "const int src_y = min(" not in cu_source
    assert "const int c = max(" not in cu_source
    assert '\\"zero_copy\\":true' in cu_source
    assert '\\"memory_space\\":\\"cuda_device' in cu_source
    assert "release_token" in cu_source
    assert "novasight_release_tensor" in cu_source


def test_native_jetson_support_escapes_diagnostic_json() -> None:
    support = Path(
        "novasight_jetson_preprocess_native/native/src/jetson/"
        "novasight_jetson_preprocess_native_jetson_support.cpp"
    ).read_text(encoding="utf-8")

    assert "std::string json_escape" in support
    assert '"\\\\n"' in support
    assert '"\\\\\\""' in support
    assert '"{\\"reason\\":\\"" + json_escape(reason)' in support
    assert '"\\"detail\\":\\"" + json_escape(detail)' in support
    assert '"\\"backend\\":\\"" + json_escape(backend)' in support


def test_native_jetson_support_validates_required_payload_contract() -> None:
    support = Path(
        "novasight_jetson_preprocess_native/native/src/jetson/"
        "novasight_jetson_preprocess_native_jetson_support.cpp"
    ).read_text(encoding="utf-8")
    reference = Path(
        "novasight_jetson_preprocess_native/native/src/"
        "novasight_jetson_preprocess_native.cpp"
    ).read_text(encoding="utf-8")

    for source in (support, reference):
        assert "json.find(token)" not in source
        assert "depth == 1 && field_name == key_text" in source
        assert "bool json_value_terminator" in source
        assert "!json_value_terminator(*end)" in source
        assert "!json_value_terminator(cursor[1])" in source
        assert "json_value_terminator(cursor[4])" in source
        assert "json_value_terminator(cursor[5])" in source
        assert "bool expect_value = true" in source
        assert 'capture_ts_ns"' in source
        assert 'dmabuf_fd"' in source
        assert 'resource_source' in source
        assert 'resource_width' in source
        assert 'resource_height' in source
        assert 'resource_pixel_format' in source
        assert 'resource_geometry_mismatch' in source
        assert 'resource_format_mismatch' in source
        assert "appsink" in source
        assert 'pixel_format' in source
        assert "NV12" in source
        assert 'roi_offset_x' in source
        assert 'roi_offset_y' in source
        assert "invalid_roi_offset" in source
        assert 'needs_resize' in source
        assert "needs_resize_required" in source
        assert 'read_object_field(json, "model_shape", &model_shape_json)' in source
        assert 'for (const char* key : {"batch", "channels", "height", "width"})' in source
        assert 'Expected positive integer field: model_shape.' in source
        assert 'read_int_array_field(json, "nchw"' in source


def test_runtime_config_defaults_are_stable() -> None:
    cfg = RuntimeConfig()

    assert cfg.web.port == 5174
    assert cfg.hardware.auto_connect is True
    assert cfg.roi.size == 640
    assert cfg.roi.mode == "center"
    assert cfg.calibration.profile_id == "default"
    assert cfg.calibration.profile_version == 1
    assert cfg.calibration.game_sensitivity_fingerprint == "unverified-default"
    # Old attributes that drove the first prototype must not have leaked back.
    assert not hasattr(cfg, "model_path")
    assert not hasattr(cfg, "plugin_settings")


def test_capture_gstreamer_candidates_have_upstream_latest_only_queue() -> None:
    profile = CaptureProfile(
        device="/dev/video0",
        width=1920,
        height=1080,
        fps=120,
        pixel_format="MJPG",
        preference="manual",
        selection_reason="test",
    )

    candidates = [
        *build_pipeline_candidates(profile),
        *build_appsink_candidates(profile, roi_size=640),
        *build_resource_appsink_candidates(profile, roi_size=640),
    ]

    assert candidates
    for candidate in candidates:
        assert LATEST_ONLY_QUEUE in candidate.pipeline
        assert "appsink" in candidate.pipeline
        assert "max-buffers=1" in candidate.pipeline
        assert "drop=true" in candidate.pipeline
        assert candidate.pipeline.index(LATEST_ONLY_QUEUE) < candidate.pipeline.rindex("appsink")


def test_nvmm_resource_candidates_include_decoupled_preview_branch() -> None:
    profile = CaptureProfile(
        device="/dev/video0",
        width=1920,
        height=1080,
        fps=120,
        pixel_format="MJPG",
        preference="manual",
        selection_reason="test",
    )

    candidates = build_resource_appsink_candidates(profile, roi_size=640)

    assert candidates
    for candidate in candidates:
        assert candidate.pipeline.count("v4l2src") == 1
        assert "tee name=novasight_preview_split" in candidate.pipeline
        assert "appsink name=sink" in candidate.pipeline
        assert "appsink name=preview_sink" in candidate.pipeline
        assert "video/x-raw,format=BGRx,width=640,height=640" in candidate.pipeline


def test_runtime_config_defaults_include_exclusive_dual_mouse_control_settings() -> None:
    cfg = RuntimeConfig()

    assert cfg.control.mode == "universal_saturated"
    assert cfg.control.aim.y_ratio == 0.22
    assert cfg.control.configured_actuation_delay_s == 0.004
    assert cfg.control.prediction_strength == 1.0
    assert cfg.control.prediction_x_enabled is True
    assert cfg.control.prediction_y_enabled is True
    assert cfg.control.calibrated_angular.fov_x_deg == 105.0
    assert cfg.control.calibrated_angular.counts_per_360_x == 9980.0
    assert cfg.control.calibrated_angular.kp_x == 1.0
    assert cfg.control.calibrated_angular.kd_x == 0.0
    assert cfg.control.calibrated_angular.d_ema_alpha == 0.30
    assert cfg.control.universal_saturated.response_scale_x_px == 160.0
    assert cfg.control.universal_saturated.max_step_x_counts == 30.0
    assert cfg.control.shared.deadzone_x_px == 0.0
    assert cfg.control.shared.max_count_slew_x == 10.0
    assert cfg.control.shared.invert_y is False
    assert cfg.control.scheduler_step_counts_x == 20
    assert cfg.control.scheduler_step_counts_y == 20
    assert cfg.control.scheduler_interval_ms == 4.0
    assert cfg.control.target_fov_radius_px == 180.0
    assert cfg.control.min_confidence == 0.25
    assert cfg.control.target_switch_delay_ms == 50.0
    assert cfg.control.tracker_max_match_distance == 1.5
    assert cfg.control.tracker_position_cost_weight == 0.75
    assert cfg.control.tracker_iou_cost_weight == 0.25
    assert cfg.control.tracker_max_missed_frames == 2
    assert cfg.control.target_switch_min_preference_advantage == 0.08
    assert cfg.control.target_switch_min_continuity_score == 0.70
    assert cfg.control.kalman_acceleration_noise == 1200.0
    assert cfg.control.kalman_measurement_noise_x == 16.0
    assert cfg.control.kalman_measurement_noise_y == 16.0
    assert cfg.control.kalman_max_predict_missing_ms == 80.0
    assert cfg.control.kalman_max_predict_steps == 5
    assert cfg.control.kalman_max_predict_dt_ms == 35.0
    assert cfg.control.kalman_max_position_sigma_px == 45.0
    assert cfg.control.kalman_max_covariance_trace == 5000.0
    assert cfg.control.kalman_nis_threshold == 9.21
    assert cfg.control.kalman_nis_hard_reject == 16.0
    assert cfg.control.kalman_min_identity_confidence == 0.70
    assert cfg.control.kalman_min_prediction_confidence == 0.35
    assert cfg.control.kalman_prediction_decay_tau_ms == 45.0


def test_runtime_config_defaults_include_inference_settings() -> None:
    cfg = RuntimeConfig()

    assert cfg.capture.backend == "gst_cpu_latest"
    assert cfg.capture.memory == "system"
    assert cfg.capture.latest_only is True
    assert cfg.capture.appsink_max_buffers == 1
    assert cfg.capture.queue_leaky == "downstream"
    assert cfg.preprocess.backend == "cpu"
    assert cfg.preprocess.input_format == "auto"
    assert cfg.preprocess.output_dtype == "fp16"
    assert cfg.preprocess.normalize is True
    assert cfg.preprocess.use_pinned_memory is True
    assert cfg.preprocess.h2d_async is True
    assert cfg.inference.enabled is True
    assert cfg.inference.backend == "tensorrt"
    assert cfg.inference.device == "cuda"
    assert cfg.inference.require_gpu is True
    assert cfg.inference.allow_cpu_fallback is False
    assert cfg.inference.inference_input_deadline_ms == 55.0
    assert cfg.inference.confidence_threshold == 0.25
    assert cfg.inference.nms_threshold == 0.45
    assert cfg.inference.input_source == "source.default"
    assert cfg.runtime.freshness_threshold_ms == 55.0
    assert cfg.runtime.drop_stale_batches is True
    assert cfg.runtime.consume_latest_only is True


def test_runtime_config_migrates_legacy_full_deepstream_keys() -> None:
    cfg = parse_runtime_config(
        {
            "capture": {"memory": "cpu"},
            "inference": {
                "backend": "deepstream",
                "deepstream_manifest_path": "combat/default/model.manifest.json",
                "deepstream_config_path": "combat/default/deepstream.ini",
                "deepstream_io_mode": 4,
                "deepstream_batched_push_timeout_us": 12000,
            }
        }
    )

    assert cfg.source.default == "null"
    assert cfg.capture.backend == "gst_cpu_latest"
    assert cfg.capture.memory == "system"
    assert cfg.preprocess.backend == "cpu"
    assert cfg.inference.backend == "tensorrt"
    assert not hasattr(cfg.inference, "deepstream_manifest_path")
    assert not hasattr(cfg.inference, "deepstream_config_path")
    assert not hasattr(cfg.inference, "deepstream_io_mode")
    assert not hasattr(cfg.inference, "deepstream_batched_push_timeout_us")


def test_runtime_config_accepts_tensorrt_and_nvmm_latest_backends() -> None:
    cfg = parse_runtime_config({"inference": {"backend": "tensorrt"}})

    assert cfg.inference.backend == "tensorrt"

    cfg = parse_runtime_config(
        {
            "capture": {"backend": "nvmm_latest", "memory": "nvmm"},
            "preprocess": {"backend": "cuda"},
            "inference": {"backend": "nvmm_latest"},
        }
    )

    assert cfg.inference.backend == "nvmm_latest"


def test_runtime_config_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "novasight.yaml"
    cfg = RuntimeConfig()
    cfg.web.port = 6000
    cfg.source.default = "image:/tmp/frame.jpg"
    cfg.calibration.profile_id = "arena-105"
    cfg.control.calibrated_angular.counts_per_360_y = 10010

    save_runtime_config(cfg, path)
    loaded = load_runtime_config(path)

    assert loaded.web.port == 6000
    assert loaded.source.default == "image:/tmp/frame.jpg"
    assert loaded.calibration.profile_id == "arena-105"
    assert loaded.control.calibrated_angular.counts_per_360_y == 10010


@pytest.mark.parametrize(
    "yaml_text",
    ["", "null\n", "   \n"],
)
def test_runtime_config_returns_defaults_for_blank_or_empty_inputs(
    tmp_path: Path, yaml_text: str
) -> None:
    path = tmp_path / "novasight.yaml"
    path.write_text(yaml_text, encoding="utf-8")

    assert load_runtime_config(path) == RuntimeConfig()


def test_runtime_config_missing_file_returns_defaults(tmp_path: Path) -> None:
    assert load_runtime_config(tmp_path / "missing.yaml") == RuntimeConfig()


def test_example_runtime_config_loads_with_current_schema() -> None:
    cfg = load_runtime_config(Path("config/novasight.example.yaml"))

    assert cfg.source.default == "capture"
    assert cfg.capture.backend == "nvmm_latest"
    assert cfg.capture.memory == "nvmm"
    assert cfg.preprocess.backend == "cuda"
    assert cfg.inference.backend == "nvmm_latest"
    assert cfg.consumers.inference is True
    assert cfg.consumers.recording_format == "csv"
    assert cfg.control.mode == "universal_saturated"
    assert cfg.control.calibrated_angular.fov_x_deg == 105
    assert cfg.control.calibrated_angular.counts_per_360_x == 9980
    assert cfg.control.shared.invert_y is False
    assert cfg.hardware.auto_connect is True
    assert cfg.control.aim.y_ratio == pytest.approx(0.22)
    assert cfg.control.configured_actuation_delay_s == pytest.approx(0.004)


def test_runtime_config_migrates_legacy_axis_signs() -> None:
    cfg = parse_runtime_config(
        {"calibration": {"axis_sign_x": 1, "axis_sign_y": -1}}
    )

    assert cfg.control.shared.invert_y is True
    assert not hasattr(cfg.calibration, "axis_sign_x")
    assert not hasattr(cfg.calibration, "axis_sign_y")


def test_runtime_config_drops_legacy_noop_hardware_flip_dy() -> None:
    cfg = parse_runtime_config({"hardware": {"flip_dy": True}})

    assert cfg.control.shared.invert_y is False
    assert not hasattr(cfg.hardware, "flip_dy")


def test_runtime_config_rejects_unrepresentable_legacy_x_inversion() -> None:
    with pytest.raises(ValueError, match="axis_sign_x=-1.*cannot be migrated"):
        parse_runtime_config(
            {"calibration": {"axis_sign_x": -1, "axis_sign_y": 1}}
        )


def test_runtime_config_migrates_previous_mouse_control_schema() -> None:
    cfg = parse_runtime_config(
        {
            "runtime": {"freshness_threshold_ms": 60.0},
            "calibration": {"axis_sign_x": 1, "axis_sign_y": -1},
            "control": {
                "min_confidence": 0.0,
                "aim_ratio": 40.0,
                "configured_extra_prediction_delay_ms": 2.0,
                "latency_compensation_enabled": True,
                "latency_compensation_scale": 0.7,
                "latency_reject_if_age_exceeds_ms": 55.0,
                "strategy": "experimental_angle_pid",
                "command_interval_ms": 4.0,
                "scheduler_max_step_x": 12,
                "scheduler_max_step_y": 10,
                "experimental_angle_fov_x_deg": 103.0,
                "experimental_angle_counts_per_360": 9900.0,
                "experimental_angle_kp_x": 0.4,
                "experimental_angle_kp_y": 0.3,
                "experimental_angle_kd": 0.02,
                "experimental_angle_derivative_filter": 0.25,
                "experimental_angle_deadzone_px": 1.5,
                "experimental_angle_max_control_angle_deg": 3.0,
                "experimental_angle_magnet_enabled": False,
                "move_kind": "bezier",
                "move_ms": 12,
            },
        }
    )

    assert cfg.control.mode == "calibrated_angular"
    assert cfg.control.calibrated_angular.fov_x_deg == pytest.approx(103.0)
    assert cfg.control.calibrated_angular.counts_per_360_x == pytest.approx(9900.0)
    assert cfg.control.calibrated_angular.counts_per_360_y == pytest.approx(9900.0)
    assert cfg.control.shared.invert_y is True
    assert cfg.control.min_confidence == pytest.approx(0.10)
    assert cfg.control.aim.y_ratio == pytest.approx(0.40)
    assert cfg.control.configured_actuation_delay_s == pytest.approx(0.002)
    assert cfg.control.prediction_strength == pytest.approx(0.7)
    assert cfg.control.calibrated_angular.kp_x == pytest.approx(0.4)
    assert cfg.control.calibrated_angular.kp_y == pytest.approx(0.3)
    assert cfg.control.calibrated_angular.kd_x == pytest.approx(0.02)
    assert cfg.control.calibrated_angular.kd_y == pytest.approx(0.02)
    assert cfg.control.calibrated_angular.d_ema_alpha == pytest.approx(0.25)
    assert cfg.control.shared.deadzone_x_px == pytest.approx(1.5)
    assert cfg.control.shared.deadzone_y_px == pytest.approx(1.5)
    assert cfg.control.calibrated_angular.max_angle_step_x_deg == pytest.approx(3.0)
    assert cfg.control.calibrated_angular.max_angle_step_y_deg == pytest.approx(3.0)
    assert cfg.control.scheduler_interval_ms == pytest.approx(4.0)
    assert cfg.control.scheduler_step_counts_x == 12
    assert cfg.control.scheduler_step_counts_y == 10
    assert cfg.runtime.freshness_threshold_ms == pytest.approx(60.0)


def test_runtime_config_validates_recording_format() -> None:
    assert parse_runtime_config({"consumers": {"recording_format": "csv"}}).consumers.recording_format == "csv"
    assert (
        parse_runtime_config({"consumers": {"recording_format": "parquet"}}).consumers.recording_format
        == "parquet"
    )

    with pytest.raises(ValueError, match="consumers.recording_format.*csv or parquet"):
        parse_runtime_config({"consumers": {"recording_format": "jsonl"}})


@pytest.mark.parametrize(
    ("raw", "key_path"),
    [
        ({"calibration": {"profile_id": ""}}, "calibration.profile_id"),
        ({"calibration": {"profile_version": 0}}, "calibration.profile_version"),
        ({"calibration": {"game_sensitivity_fingerprint": ""}}, "calibration.game_sensitivity_fingerprint"),
        ({"control": {"mode": "mixed"}}, "control.mode"),
        ({"control": {"calibrated_angular": {"fov_x_deg": 180}}}, "control.calibrated_angular.fov_x_deg"),
        ({"control": {"calibrated_angular": {"counts_per_360_x": 0}}}, "control.calibrated_angular.counts_per_360_x"),
        ({"control": {"calibrated_angular": {"counts_per_360_y": 0}}}, "control.calibrated_angular.counts_per_360_y"),
        ({"control": {"configured_actuation_delay_s": -0.001}}, "control.configured_actuation_delay_s"),
        ({"control": {"prediction_strength": 1.51}}, "control.prediction_strength"),
        ({"control": {"calibrated_angular": {"kp_x": 2.01}}}, "control.calibrated_angular.kp_x"),
        ({"control": {"calibrated_angular": {"kd_y": -0.01}}}, "control.calibrated_angular.kd_y"),
        ({"control": {"calibrated_angular": {"d_ema_alpha": 0.0}}}, "control.calibrated_angular.d_ema_alpha"),
        ({"control": {"calibrated_angular": {"max_angle_step_x_deg": 0.0}}}, "control.calibrated_angular.max_angle_step_x_deg"),
        ({"control": {"universal_saturated": {"response_scale_x_px": 0.0}}}, "control.universal_saturated.response_scale_x_px"),
        ({"control": {"universal_saturated": {"max_step_y_counts": 0.0}}}, "control.universal_saturated.max_step_y_counts"),
        ({"control": {"shared": {"deadzone_x_px": 10.1}}}, "control.shared.deadzone_x_px"),
        ({"control": {"shared": {"max_count_slew_y": 0.0}}}, "control.shared.max_count_slew_y"),
        ({"control": {"scheduler_interval_ms": 0.1}}, "control.scheduler_interval_ms"),
        ({"control": {"scheduler_step_counts_x": 0}}, "control.scheduler_step_counts_x"),
        ({"control": {"scheduler_step_counts_y": 21}}, "control.scheduler_step_counts_y"),
        ({"control": {"target_fov_radius_px": 0}}, "control.target_fov_radius_px"),
        ({"control": {"min_confidence": 0.09}}, "control.min_confidence"),
        ({"control": {"target_switch_delay_ms": 501}}, "control.target_switch_delay_ms"),
        ({"control": {"tracker_max_match_distance": 0}}, "control.tracker_max_match_distance"),
        ({"control": {"tracker_position_cost_weight": -0.1}}, "control.tracker_position_cost_weight"),
        ({"control": {"tracker_iou_cost_weight": -0.1}}, "control.tracker_iou_cost_weight"),
        ({"control": {"tracker_max_missed_frames": -1}}, "control.tracker_max_missed_frames"),
        ({"control": {"target_switch_min_preference_advantage": -0.1}}, "control.target_switch_min_preference_advantage"),
        ({"control": {"target_switch_min_continuity_score": 1.5}}, "control.target_switch_min_continuity_score"),
        ({"control": {"kalman_max_predict_steps": 0}}, "control.kalman_max_predict_steps"),
        ({"control": {"kalman_min_prediction_confidence": 1.5}}, "control.kalman_min_prediction_confidence"),
        ({"control": {"kalman_nis_hard_reject": 8, "kalman_nis_threshold": 9}}, "control.kalman_nis_hard_reject"),
    ],
)
def test_runtime_config_rejects_invalid_calibration(raw: dict[str, object], key_path: str) -> None:
    with pytest.raises(ValueError, match=key_path):
        parse_runtime_config(raw)


def test_runtime_config_has_one_kmnet_runtime_path() -> None:
    assert parse_runtime_config({"control": {"trigger_mode": "hardware"}}).control.trigger_mode == "hardware"
    assert parse_runtime_config({"control": {"trigger_mode": "always"}}).control.trigger_mode == "always"

    with pytest.raises(ValueError, match="control.trigger_mode.*hardware or always"):
        parse_runtime_config({"control": {"trigger_mode": "telemetry"}})
    migrated = parse_runtime_config(
        {
            "control": {
                "output_mode": "dry_run",
                "lost_target_timeout_ms": 120,
                "tracker_confirm_frames": 2,
                "tracker_matching_distance_px": 140,
            },
            "executor": {"default": "silent"},
            "hardware": {"kind": "makcu", "serial_port": "/dev/ttyUSB0"},
        }
    )
    assert not hasattr(migrated.control, "output_mode")
    assert not hasattr(migrated.control, "lost_target_timeout_ms")
    assert not hasattr(migrated.control, "tracker_confirm_frames")
    assert not hasattr(migrated.control, "tracker_matching_distance_px")
    assert not hasattr(migrated, "executor")
    assert not hasattr(migrated.hardware, "kind")
    assert not hasattr(migrated.hardware, "serial_port")
    with pytest.raises(ValueError, match="control.strategy.*experimental_angle_pid"):
        parse_runtime_config({"control": {"strategy": "pid"}})


def test_runtime_config_partial_nested_config_preserves_defaults(
    tmp_path: Path,
) -> None:
    path = tmp_path / "novasight.yaml"
    path.write_text("web:\n  port: 6000\n", encoding="utf-8")

    loaded = load_runtime_config(path)

    assert loaded.web.port == 6000
    assert loaded.web.host == "0.0.0.0"


def test_save_runtime_config_creates_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "config" / "novasight.yaml"

    save_runtime_config(RuntimeConfig(), path)

    assert path.exists()


def test_runtime_config_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    path = tmp_path / "novasight.yaml"
    path.write_text("false\n", encoding="utf-8")

    with pytest.raises(ValueError, match="runtime config must be a mapping"):
        load_runtime_config(path)


def test_runtime_config_rejects_nested_non_mapping_section(tmp_path: Path) -> None:
    path = tmp_path / "novasight.yaml"
    path.write_text("web: false\n", encoding="utf-8")

    with pytest.raises(ValueError, match="web.*must be a mapping"):
        load_runtime_config(path)


@pytest.mark.parametrize("unknown_key", ["model_registry", "weeb"])
def test_runtime_config_rejects_unknown_top_level_key(
    tmp_path: Path, unknown_key: str
) -> None:
    path = tmp_path / "novasight.yaml"
    path.write_text(f"{unknown_key}: {{}}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=f"unknown config key.*{unknown_key}"):
        load_runtime_config(path)


def test_runtime_config_rejects_unknown_nested_key(tmp_path: Path) -> None:
    path = tmp_path / "novasight.yaml"
    path.write_text("web:\n  prt: 5174\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown config key.*web.prt"):
        load_runtime_config(path)


def test_runtime_config_rejects_removed_max_frame_queue(tmp_path: Path) -> None:
    path = tmp_path / "novasight.yaml"
    path.write_text("limits:\n  max_frame_queue: 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown config key.*limits.max_frame_queue"):
        load_runtime_config(path)


@pytest.mark.parametrize(
    "legacy_key",
    [
        "pid_kp_x",
        "kp_x_move_max",
        "prediction_factor",
        "y_down_enabled",
        "isolated_kp_x",
        "dynamic_pid_kp_x",
        "trigger_bindings",
        "deadzone_counts",
        "near_px",
        "near_speed",
        "far_speed",
        "ema_alpha",
        "counts_per_revolution_x",
        "counts_per_revolution_y",
    ],
)
def test_runtime_config_rejects_removed_control_fields(legacy_key: str) -> None:
    with pytest.raises(ValueError, match=f"unknown config key.*control.{legacy_key}"):
        parse_runtime_config({"control": {legacy_key: 1}})


@pytest.mark.parametrize(
    ("yaml_text", "key_path"),
    [
        ("web:\n  port: nope\n", "web.port"),
        ("source:\n  target_fps: []\n", "source.target_fps"),
        ("source:\n  default: null\n", "source.default"),
    ],
)
def test_runtime_config_rejects_invalid_leaf_types(
    tmp_path: Path, yaml_text: str, key_path: str
) -> None:
    path = tmp_path / "novasight.yaml"
    path.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(ValueError, match=key_path):
        load_runtime_config(path)


def test_runtime_config_restricts_preview_fps_to_supported_values() -> None:
    for fps in (15, 30, 60):
        cfg = parse_runtime_config({"limits": {"stream_fps": fps}})
        assert cfg.limits.stream_fps == fps

    with pytest.raises(ValueError, match="limits.stream_fps.*15, 30, 60"):
        parse_runtime_config({"limits": {"stream_fps": 120}})


def test_runtime_config_restricts_roi_to_supported_center_sizes() -> None:
    for size in (640, 480, 320, 256):
        cfg = parse_runtime_config({"roi": {"size": size}})
        assert cfg.roi.size == size
        assert cfg.roi.mode == "center"

    with pytest.raises(ValueError, match="unsupported ROI size"):
        parse_runtime_config({"roi": {"size": 512}})

    cfg = parse_runtime_config({"roi": {"mode": "manual"}})
    assert cfg.roi.mode == "manual"


def test_runtime_config_capture_memory_matches_backend() -> None:
    assert parse_runtime_config({}).capture.memory == "system"
    assert parse_runtime_config({"capture": {"memory": "cpu"}}).capture.memory == "system"
    assert parse_runtime_config(
        {"capture": {"backend": "nvmm_latest", "memory": "nvmm"}, "preprocess": {"backend": "cuda"}, "inference": {"backend": "nvmm_latest"}}
    ).capture.memory == "nvmm"

    with pytest.raises(ValueError, match="capture.memory.*system or nvmm"):
        parse_runtime_config({"capture": {"memory": "dmabuf"}})


def test_runtime_config_schema_exposes_roi_size() -> None:
    schema = runtime_config_schema(RuntimeConfig())
    roi_section = next(section for section in schema["sections"] if section["id"] == "roi")

    fields = {field["path"]: field for field in roi_section["fields"]}

    assert fields["roi.size"] == {
        "path": "roi.size",
        "label": "中心 ROI",
        "type": "select",
        "options": ["640", "480", "320", "256"],
        "restart_required": False,
    }
    assert {"roi.offset_x", "roi.offset_y"}.issubset(fields)


def test_runtime_config_schema_exposes_capture_memory() -> None:
    schema = runtime_config_schema(RuntimeConfig())
    capture_section = next(section for section in schema["sections"] if section["id"] == "capture")
    fields = {field["path"]: field for field in capture_section["fields"]}

    assert fields["capture.memory"] == {
        "path": "capture.memory",
        "label": "采集内存路径",
        "type": "select",
        "options": ["system", "nvmm"],
        "restart_required": True,
    }
    assert fields["capture.backend"]["options"] == ["gst_cpu_latest", "nvmm_latest"]


def test_runtime_config_schema_exposes_only_exclusive_dual_mouse_control_fields() -> None:
    schema = runtime_config_schema(RuntimeConfig())
    control_section = next(section for section in schema["sections"] if section["id"] == "control")
    paths = {field["path"] for field in control_section["fields"]}

    assert {
        "control.trigger_mode",
        "control.mode",
        "control.aim.y_ratio",
        "control.configured_actuation_delay_s",
        "control.prediction_strength",
        "control.prediction_x_enabled",
        "control.prediction_y_enabled",
        "control.calibrated_angular.fov_x_deg",
        "control.calibrated_angular.counts_per_360_x",
        "control.calibrated_angular.counts_per_360_y",
        "control.calibrated_angular.kp_x",
        "control.calibrated_angular.kp_y",
        "control.calibrated_angular.kd_x",
        "control.calibrated_angular.kd_y",
        "control.calibrated_angular.d_ema_alpha",
        "control.calibrated_angular.max_angle_step_x_deg",
        "control.calibrated_angular.max_angle_step_y_deg",
        "control.universal_saturated.response_scale_x_px",
        "control.universal_saturated.response_scale_y_px",
        "control.universal_saturated.max_step_x_counts",
        "control.universal_saturated.max_step_y_counts",
        "control.shared.deadzone_x_px",
        "control.shared.deadzone_y_px",
        "control.shared.max_count_slew_x",
        "control.shared.max_count_slew_y",
        "control.shared.invert_y",
        "control.scheduler_step_counts_x",
        "control.scheduler_step_counts_y",
        "control.scheduler_interval_ms",
        "control.target_fov_radius_px",
        "control.min_confidence",
        "control.target_switch_delay_ms",
    }.issubset(paths)
    assert "control.pid_kp_x" not in paths
    assert "control.pid_kp_y" not in paths
    assert "control.pid_ki_x" not in paths
    assert "control.pid_kd_x" not in paths
    assert "control.kp_x_move_max" not in paths
    assert "control.kp_y_move_max" not in paths
    assert "control.dynamic_pid_kp_x" not in paths
    assert "control.isolated_kp_x" not in paths
    assert "control.y_down_enabled" not in paths
    assert "control.y_rate_window_ms" not in paths
    assert "control.deadzone_counts" not in paths
    assert "control.near_px" not in paths
    assert "control.near_speed" not in paths
    assert "control.far_speed" not in paths
    assert "control.ema_alpha" not in paths
    assert "control.counts_per_revolution_x" not in paths
    assert "control.counts_per_revolution_y" not in paths
    assert "control.trigger_bindings" not in paths
    assert "control.experimental_angle_fov_x_deg" not in paths
    assert "control.experimental_angle_counts_per_360" not in paths
    assert "control.experimental_angle_sign_x" not in paths
    assert "control.experimental_angle_sign_y" not in paths
    assert not any(path.startswith("control.experimental_angle_") for path in paths)
    control_values = schema["values"]["control"]
    assert "pid_kp_x" not in control_values
    assert "dynamic_pid_kp_x" not in control_values
    assert "isolated_kp_x" not in control_values
    assert "y_down_enabled" not in control_values
    assert "deadzone_counts" not in control_values
    assert "near_px" not in control_values
    assert "near_speed" not in control_values
    assert "far_speed" not in control_values
    assert "ema_alpha" not in control_values
    assert "counts_per_revolution_x" not in control_values
    assert "counts_per_revolution_y" not in control_values
    assert "strategy" not in control_values
    assert "control.configured_extra_prediction_delay_ms" not in paths
    assert "control.latency_estimated_actuation_delay_ms" not in paths
    assert "control.kp_x" not in paths
    assert "control.deadzone_px_x" not in paths


def test_runtime_config_schema_exposes_calibration_profile() -> None:
    schema = runtime_config_schema(RuntimeConfig())
    calibration_section = next(section for section in schema["sections"] if section["id"] == "calibration")
    paths = {field["path"] for field in calibration_section["fields"]}

    assert {
        "calibration.profile_id",
        "calibration.profile_version",
        "calibration.game_sensitivity_fingerprint",
    }.issubset(paths)
    assert "calibration.fov_x_deg" not in paths
    assert "calibration.counts_per_360_x" not in paths


def test_runtime_config_schema_exposes_kmnet_auto_connect() -> None:
    schema = runtime_config_schema(RuntimeConfig())
    hardware_section = next(section for section in schema["sections"] if section["id"] == "hardware")
    fields = {field["path"]: field for field in hardware_section["fields"]}

    assert fields["hardware.auto_connect"] == {
        "path": "hardware.auto_connect",
        "label": "服务启动自动连接",
        "type": "bool",
        "restart_required": False,
    }


def test_runtime_config_schema_exposes_editable_inference_fields() -> None:
    schema = runtime_config_schema(RuntimeConfig())
    inference_section = next(section for section in schema["sections"] if section["id"] == "inference")
    paths = {field["path"] for field in inference_section["fields"]}

    assert {
        "inference.enabled",
        "inference.backend",
        "inference.device",
        "inference.require_gpu",
        "inference.allow_cpu_fallback",
        "inference.inference_input_deadline_ms",
        "inference.confidence_threshold",
        "inference.nms_threshold",
        "inference.input_source",
    }.issubset(paths)
    backend_field = next(
        field for field in inference_section["fields"] if field["path"] == "inference.backend"
    )
    assert backend_field["options"] == ["tensorrt", "nvmm_latest"]


def test_runtime_config_schema_exposes_latest_only_runtime_fields() -> None:
    schema = runtime_config_schema(RuntimeConfig())
    runtime_section = next(section for section in schema["sections"] if section["id"] == "runtime")
    paths = {field["path"] for field in runtime_section["fields"]}

    assert {
        "runtime.freshness_threshold_ms",
        "runtime.drop_stale_batches",
        "runtime.consume_latest_only",
    }.issubset(paths)
