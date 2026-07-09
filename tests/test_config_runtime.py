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
    assert "nvbufsurface.h" in cmake
    assert "nvbufsurftransform" in cmake
    assert "NOVASIGHT_JETSON_PREPROCESS_SUPPORT_SOURCES" in cmake
    assert "novasight_jetson_preprocess_native_jetson_support.cpp" in cmake
    assert "${CMAKE_CURRENT_LIST_DIR}/src/jetson" in cmake
    assert "OUTPUT_NAME novasight_preprocess" in cmake
    assert "jetson_cuda_preprocess_not_compiled" not in cmake


def test_native_jetson_cuda_source_uses_real_dmabuf_egl_cuda_path() -> None:
    source = Path(
        "novasight_jetson_preprocess_native/native/src/jetson/"
        "novasight_jetson_preprocess_native_jetson_cuda.cu"
    ).read_text(encoding="utf-8")

    assert "NvBufSurfaceFromFd" in source
    assert "#include <nvbufsurftransform.h>" in source
    assert "NvBufSurfTransform" in source
    assert "NVBUF_COLOR_FORMAT_RGBA" in source
    assert "rgba_to_nchw_kernel" in source
    assert '\\"timings\\"' in source
    assert '\\"nvbufsurftransform_ms\\"' in source
    assert '\\"egl_cuda_map_ms\\"' in source
    assert '\\"cuda_kernel_ms\\"' in source
    assert "NvBufSurfaceMapEglImage" in source
    assert "cuGraphicsEGLRegisterImage" in source
    assert "cuGraphicsResourceGetMappedEglFrame" in source
    assert "cudaMemcpy2DFromArrayAsync" in source
    assert "surface->surfaceList[0].width" in source
    assert "surface->surfaceList[0].height" in source
    assert "nvbufsurface_geometry_mismatch" in source
    assert "bool validate_rgba_plane_layout" in source
    assert "cuda_egl_rgba_plane_invalid" in source
    assert "rgba_pitch < width * 4" in source
    assert "__device__ int scaled_source_index" in source
    assert "nv12_to_nchw_kernel" not in source
    assert "max(0, min(255" not in source
    assert "const int src_x = min(" not in source
    assert "const int src_y = min(" not in source
    assert "const int c = max(" not in source
    assert '\\"zero_copy\\":true' in source
    assert '\\"memory_space\\":\\"cuda_device' in source
    assert "release_token" in source
    assert "novasight_release_tensor" in source


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
    assert cfg.executor.default == "kmnet"
    assert cfg.control.output_mode == "kmnet"
    assert cfg.control.strategy == "experimental_angle_pid"
    assert cfg.hardware.kind == "kmnet"
    assert cfg.roi.size == 640
    assert cfg.roi.mode == "center"
    assert cfg.calibration.profile_id == "default"
    assert cfg.calibration.profile_version == 1
    assert cfg.calibration.fov_semantics == "horizontal"
    assert cfg.calibration.fov_x_deg == 105.0
    assert cfg.calibration.counts_per_360_x == 9980.0
    assert cfg.calibration.counts_per_360_y == 9980.0
    assert cfg.calibration.axis_sign_x == 1.0
    assert cfg.calibration.axis_sign_y == 1.0
    assert cfg.calibration.game_sensitivity_fingerprint == "unverified-default"
    assert cfg.calibration.projection_profile == "fixed_horizontal_fov"
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


def test_runtime_config_defaults_include_experimental_angle_settings() -> None:
    cfg = RuntimeConfig()

    assert cfg.control.experimental_angle_kp_x == 0.35
    assert cfg.control.experimental_angle_kp_y == 0.24
    assert cfg.control.experimental_angle_ki == 0.0
    assert cfg.control.experimental_angle_kd == 0.0
    assert cfg.control.experimental_angle_integral_limit == 0.0
    assert cfg.control.experimental_angle_near_error_deg == 0.35
    assert cfg.control.experimental_angle_far_error_deg == 2.50
    assert cfg.control.experimental_angle_near_kp_scale == 0.35
    assert cfg.control.experimental_angle_middle_kp_scale == 0.70
    assert cfg.control.experimental_angle_far_kp_scale == 1.00
    assert cfg.control.experimental_angle_near_kd_scale == 1.00
    assert cfg.control.experimental_angle_middle_kd_scale == 0.80
    assert cfg.control.experimental_angle_far_kd_scale == 0.50
    assert cfg.control.experimental_angle_prediction_gain_min == 0.35
    assert cfg.control.experimental_angle_prediction_d_gain_min == 0.25
    assert cfg.control.experimental_angle_max_control_angle_deg == 3.0
    assert cfg.control.experimental_angle_max_step_counts == 80.0
    assert cfg.control.experimental_angle_max_counts_delta_x == 35.0
    assert cfg.control.experimental_angle_max_counts_delta_y == 35.0
    assert cfg.control.experimental_angle_control_hz == 60.0
    assert cfg.control.command_interval_ms == 1.0
    assert cfg.control.scheduler_command_ttl_ms == 35.0
    assert cfg.control.scheduler_predicted_command_ttl_ms == 18.0
    assert cfg.control.scheduler_cancel_on_new_frame is True
    assert cfg.control.scheduler_cancel_on_direction_change is True
    assert cfg.control.scheduler_cancel_on_track_change is True
    assert cfg.control.scheduler_max_step_x == 20
    assert cfg.control.scheduler_max_step_y == 20
    assert cfg.control.scheduler_queue_hard_limit == 64
    assert cfg.control.scheduler_device_error_cooldown_ms == 50.0
    assert cfg.control.tracker_confirm_frames == 2
    assert cfg.control.target_switch_min_preference_advantage == 0.08
    assert cfg.control.target_switch_min_continuity_score == 0.70
    assert cfg.control.target_switch_confirm_frames == 3
    assert cfg.control.tracker_matching_distance_px == 140.0
    assert cfg.control.tracker_ambiguity_margin == 0.08
    assert cfg.control.tracker_missing_timeout_ms == 120.0
    assert cfg.control.tracker_delete_timeout_ms == 250.0
    assert cfg.control.tracker_match_threshold == 0.65
    assert cfg.control.tracker_mahalanobis_gate == 9.21
    assert cfg.control.kalman_enabled is True
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
    assert cfg.control.aim_horizontal_percent == 50.0
    assert cfg.control.aim_ratio == 40.0
    assert cfg.control.aim_offset_x_px == 0.0
    assert cfg.control.aim_offset_y_px == 0.0
    assert cfg.control.aim_ema_enabled is True
    assert cfg.control.aim_ema_alpha == 0.65
    assert cfg.control.aim_max_anchor_jump_ratio == 0.15
    assert cfg.control.latency_compensation_enabled is True
    assert cfg.control.latency_compensation_scale == 0.70
    assert cfg.control.latency_max_compensation_ms == 35.0
    assert cfg.control.latency_reject_if_age_exceeds_ms == 55.0
    assert cfg.control.latency_max_compensation_px == 80.0
    assert cfg.control.latency_min_velocity_px_s == 30.0
    assert cfg.control.latency_max_velocity_px_s == 2500.0
    assert cfg.control.latency_min_velocity_measurements == 3
    assert cfg.control.latency_min_velocity_confidence == 0.65
    assert cfg.control.latency_estimated_actuation_delay_ms == 2.0


def test_runtime_config_defaults_include_inference_settings() -> None:
    cfg = RuntimeConfig()

    assert cfg.capture.memory == "nvmm"
    assert cfg.inference.enabled is True
    assert cfg.inference.backend == "nvmm_latest"
    assert cfg.inference.inference_input_deadline_ms == 55.0
    assert cfg.inference.confidence_threshold == 0.25
    assert cfg.inference.nms_threshold == 0.45
    assert cfg.inference.input_source == "source.default"


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

    assert cfg.capture.memory == "nvmm"
    assert cfg.inference.backend == "nvmm_latest"
    assert not hasattr(cfg.inference, "deepstream_manifest_path")
    assert not hasattr(cfg.inference, "deepstream_config_path")
    assert not hasattr(cfg.inference, "deepstream_io_mode")
    assert not hasattr(cfg.inference, "deepstream_batched_push_timeout_us")


def test_runtime_config_accepts_nvmm_latest_backend() -> None:
    cfg = parse_runtime_config({"inference": {"backend": "nvmm_latest"}})

    assert cfg.inference.backend == "nvmm_latest"


def test_runtime_config_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "novasight.yaml"
    cfg = RuntimeConfig()
    cfg.web.port = 6000
    cfg.source.default = "image:/tmp/frame.jpg"
    cfg.calibration.profile_id = "arena-105"
    cfg.calibration.counts_per_360_y = 10010
    cfg.executor.default = "kmnet"

    save_runtime_config(cfg, path)
    loaded = load_runtime_config(path)

    assert loaded.web.port == 6000
    assert loaded.source.default == "image:/tmp/frame.jpg"
    assert loaded.calibration.profile_id == "arena-105"
    assert loaded.calibration.counts_per_360_y == 10010
    assert loaded.executor.default == "kmnet"


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

    assert cfg.inference.backend == "nvmm_latest"
    assert cfg.capture.memory == "nvmm"
    assert cfg.consumers.inference is True
    assert cfg.consumers.recording_format == "csv"
    assert cfg.calibration.fov_x_deg == 105
    assert cfg.calibration.counts_per_360_x == 9980
    assert cfg.calibration.axis_sign_x == 1


def test_runtime_config_migrates_legacy_control_calibration_fields() -> None:
    cfg = parse_runtime_config(
        {
            "control": {
                "experimental_angle_fov_x_deg": 103,
                "experimental_angle_counts_per_360": 9900,
                "experimental_angle_sign_x": -1,
                "experimental_angle_sign_y": 1,
            }
        }
    )

    assert cfg.calibration.fov_x_deg == 103
    assert cfg.calibration.counts_per_360_x == 9900
    assert cfg.calibration.counts_per_360_y == 9900
    assert cfg.calibration.axis_sign_x == -1
    assert cfg.calibration.axis_sign_y == 1


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
        ({"calibration": {"fov_semantics": "vertical"}}, "calibration.fov_semantics"),
        ({"calibration": {"fov_x_deg": 180}}, "calibration.fov_x_deg"),
        ({"calibration": {"counts_per_360_x": 0}}, "calibration.counts_per_360_x"),
        ({"calibration": {"counts_per_360_y": 0}}, "calibration.counts_per_360_y"),
        ({"calibration": {"axis_sign_x": 0}}, "calibration.axis_sign_x"),
        ({"calibration": {"axis_sign_y": 0}}, "calibration.axis_sign_y"),
        ({"calibration": {"game_sensitivity_fingerprint": ""}}, "calibration.game_sensitivity_fingerprint"),
        ({"calibration": {"projection_profile": "unknown"}}, "calibration.projection_profile"),
        ({"control": {"scheduler_command_ttl_ms": 0}}, "control.scheduler_command_ttl_ms"),
        ({"control": {"scheduler_predicted_command_ttl_ms": 0}}, "control.scheduler_predicted_command_ttl_ms"),
        ({"control": {"scheduler_max_step_x": 0}}, "control.scheduler_max_step_x"),
        ({"control": {"scheduler_max_step_y": 0}}, "control.scheduler_max_step_y"),
        ({"control": {"scheduler_queue_hard_limit": 0}}, "control.scheduler_queue_hard_limit"),
        ({"control": {"tracker_confirm_frames": 0}}, "control.tracker_confirm_frames"),
        ({"control": {"target_switch_min_preference_advantage": -0.1}}, "control.target_switch_min_preference_advantage"),
        ({"control": {"target_switch_min_continuity_score": 1.5}}, "control.target_switch_min_continuity_score"),
        ({"control": {"target_switch_confirm_frames": 0}}, "control.target_switch_confirm_frames"),
        ({"control": {"tracker_match_threshold": 1.5}}, "control.tracker_match_threshold"),
        ({"control": {"tracker_mahalanobis_gate": 0}}, "control.tracker_mahalanobis_gate"),
        ({"control": {"kalman_max_predict_steps": 0}}, "control.kalman_max_predict_steps"),
        ({"control": {"kalman_min_prediction_confidence": 1.5}}, "control.kalman_min_prediction_confidence"),
        ({"control": {"kalman_nis_hard_reject": 8, "kalman_nis_threshold": 9}}, "control.kalman_nis_hard_reject"),
        ({"control": {"aim_horizontal_percent": 101}}, "control.aim_horizontal_percent"),
        ({"control": {"aim_offset_x_px": -250}}, "control.aim_offset_x_px"),
        ({"control": {"aim_ema_alpha": 1.5}}, "control.aim_ema_alpha"),
        ({"control": {"aim_max_anchor_jump_ratio": 1.5}}, "control.aim_max_anchor_jump_ratio"),
        ({"control": {"latency_compensation_scale": 2.0}}, "control.latency_compensation_scale"),
        ({"control": {"latency_min_velocity_measurements": 0}}, "control.latency_min_velocity_measurements"),
        ({"control": {"latency_min_velocity_confidence": 1.5}}, "control.latency_min_velocity_confidence"),
        ({"control": {"latency_min_velocity_px_s": 100, "latency_max_velocity_px_s": 50}}, "control.latency_max_velocity_px_s"),
    ],
)
def test_runtime_config_rejects_invalid_calibration(raw: dict[str, object], key_path: str) -> None:
    with pytest.raises(ValueError, match=key_path):
        parse_runtime_config(raw)


def test_runtime_config_enforces_kmnet_only_runtime_paths() -> None:
    assert parse_runtime_config({"control": {"trigger_mode": "hardware"}}).control.trigger_mode == "hardware"
    assert parse_runtime_config({"control": {"trigger_mode": "always"}}).control.trigger_mode == "always"

    with pytest.raises(ValueError, match="control.trigger_mode.*hardware or always"):
        parse_runtime_config({"control": {"trigger_mode": "telemetry"}})
    with pytest.raises(ValueError, match="control.output_mode.*kmnet"):
        parse_runtime_config({"control": {"output_mode": "dry_run"}})
    with pytest.raises(ValueError, match="executor.default.*kmnet"):
        parse_runtime_config({"executor": {"default": "silent"}})
    with pytest.raises(ValueError, match="hardware.kind.*kmnet"):
        parse_runtime_config({"hardware": {"kind": "makcu"}})
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


def test_runtime_config_capture_memory_is_nvmm_only() -> None:
    assert parse_runtime_config({}).capture.memory == "nvmm"
    assert parse_runtime_config({"capture": {"memory": "cpu"}}).capture.memory == "nvmm"
    assert parse_runtime_config({"capture": {"memory": "nvmm"}}).capture.memory == "nvmm"

    with pytest.raises(ValueError, match="capture.memory.*nvmm"):
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
        "options": ["nvmm"],
        "restart_required": True,
    }


def test_runtime_config_schema_exposes_only_experimental_angle_control_fields() -> None:
    schema = runtime_config_schema(RuntimeConfig())
    control_section = next(section for section in schema["sections"] if section["id"] == "control")
    paths = {field["path"] for field in control_section["fields"]}

    assert {
        "control.trigger_mode",
        "control.output_mode",
        "control.command_interval_ms",
        "control.scheduler_command_ttl_ms",
        "control.scheduler_predicted_command_ttl_ms",
        "control.scheduler_cancel_on_new_frame",
        "control.scheduler_cancel_on_direction_change",
        "control.scheduler_cancel_on_track_change",
        "control.scheduler_max_step_x",
        "control.scheduler_max_step_y",
        "control.scheduler_queue_hard_limit",
        "control.scheduler_device_error_cooldown_ms",
        "control.experimental_angle_kp_x",
        "control.experimental_angle_kp_y",
        "control.experimental_angle_ki",
        "control.experimental_angle_kd",
        "control.experimental_angle_integral_limit",
        "control.experimental_angle_near_error_deg",
        "control.experimental_angle_far_error_deg",
        "control.experimental_angle_near_kp_scale",
        "control.experimental_angle_middle_kp_scale",
        "control.experimental_angle_far_kp_scale",
        "control.experimental_angle_near_kd_scale",
        "control.experimental_angle_middle_kd_scale",
        "control.experimental_angle_far_kd_scale",
        "control.experimental_angle_prediction_gain_min",
        "control.experimental_angle_prediction_d_gain_min",
        "control.experimental_angle_max_control_angle_deg",
        "control.experimental_angle_max_step_counts",
        "control.experimental_angle_max_counts_delta_x",
        "control.experimental_angle_max_counts_delta_y",
        "control.experimental_angle_control_hz",
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
    strategy_field = next(
        field for field in control_section["fields"] if field["path"] == "control.strategy"
    )
    assert strategy_field["options"] == ["experimental_angle_pid"]


def test_runtime_config_schema_exposes_calibration_profile() -> None:
    schema = runtime_config_schema(RuntimeConfig())
    calibration_section = next(section for section in schema["sections"] if section["id"] == "calibration")
    paths = {field["path"] for field in calibration_section["fields"]}

    assert {
        "calibration.profile_id",
        "calibration.profile_version",
        "calibration.fov_semantics",
        "calibration.fov_x_deg",
        "calibration.counts_per_360_x",
        "calibration.counts_per_360_y",
        "calibration.axis_sign_x",
        "calibration.axis_sign_y",
        "calibration.game_sensitivity_fingerprint",
        "calibration.projection_profile",
    }.issubset(paths)


def test_runtime_config_schema_exposes_editable_inference_fields() -> None:
    schema = runtime_config_schema(RuntimeConfig())
    inference_section = next(section for section in schema["sections"] if section["id"] == "inference")
    paths = {field["path"] for field in inference_section["fields"]}

    assert {
        "inference.enabled",
        "inference.backend",
        "inference.inference_input_deadline_ms",
        "inference.confidence_threshold",
        "inference.nms_threshold",
        "inference.input_source",
    }.issubset(paths)
    backend_field = next(
        field for field in inference_section["fields"] if field["path"] == "inference.backend"
    )
    assert backend_field["options"] == ["nvmm_latest"]
