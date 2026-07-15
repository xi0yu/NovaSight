from __future__ import annotations

import copy
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from novasight.config import RuntimeConfig
from novasight.config.schema import runtime_config_schema
from novasight.control.registry import (
    CALIBRATED_ANGULAR,
    DEFAULT_ACTIVE_ALGORITHM_ID,
    DUAL_PHASE_ATAN_ROBUST_PREDICTIVE_V2,
    UNIVERSAL_SATURATED,
    SchedulerPolicy,
    algorithm_definition,
    supported_algorithm_ids,
)
from novasight.executors.runtime import ExecutorRegistry
from novasight.contracts import Detection, DetectionBatch
from novasight.runtime.service import RuntimeService


def test_supported_control_algorithms_have_product_names_and_capabilities() -> None:
    assert supported_algorithm_ids() == (
        UNIVERSAL_SATURATED,
        CALIBRATED_ANGULAR,
        DUAL_PHASE_ATAN_ROBUST_PREDICTIVE_V2,
    )

    universal = algorithm_definition(UNIVERSAL_SATURATED)
    assert universal.product_name == "通用控制"
    assert universal.capabilities.requires_calibration is False
    assert universal.capabilities.supports_scheduler is True
    assert universal.capabilities.supports_prediction is False
    assert universal.capabilities.scheduler_policy is SchedulerPolicy.OPTIONAL
    assert universal.capabilities.scheduler_policy_ready is True

    calibrated = algorithm_definition(CALIBRATED_ANGULAR)
    assert calibrated.product_name == "精确角度控制"
    assert calibrated.capabilities.requires_calibration is True
    assert calibrated.capabilities.supports_scheduler is True
    assert calibrated.capabilities.supports_prediction is False
    assert calibrated.capabilities.scheduler_policy is SchedulerPolicy.CONFIGURABLE
    assert calibrated.capabilities.scheduler_policy_ready is True

    robust = algorithm_definition(DUAL_PHASE_ATAN_ROBUST_PREDICTIVE_V2)
    assert robust.product_name == "稳健预测控制"
    assert robust.capabilities.requires_calibration is True
    assert robust.capabilities.supports_scheduler is True
    assert robust.capabilities.supports_prediction is True
    assert robust.capabilities.scheduler_policy is SchedulerPolicy.LATEST_REPLACE
    assert robust.capabilities.scheduler_policy_ready is False


def test_python_runtime_default_uses_the_canonical_active_algorithm() -> None:
    assert DEFAULT_ACTIVE_ALGORITHM_ID == DUAL_PHASE_ATAN_ROBUST_PREDICTIVE_V2
    assert RuntimeConfig().control.active_algorithm == DEFAULT_ACTIVE_ALGORITHM_ID


def test_runtime_owns_only_the_active_algorithm_controller() -> None:
    config = RuntimeConfig()
    service = RuntimeService(
        config,
        models=SimpleNamespace(get_active_deployment=lambda: None),
        executors=ExecutorRegistry.from_config(config),
    )

    assert service.control_algorithms.active_algorithm_id == DEFAULT_ACTIVE_ALGORITHM_ID
    assert service.control_algorithms.active_controller is not None
    assert not hasattr(service, "mouse_controller")
    assert not hasattr(service, "dual_phase_algorithm")


@pytest.mark.parametrize(
    ("algorithm_id", "inactive_namespace"),
    [
        (UNIVERSAL_SATURATED, "calibrated_angular"),
        (CALIBRATED_ANGULAR, "universal_saturated"),
    ],
)
def test_runtime_does_not_read_inactive_algorithm_namespace(
    algorithm_id: str,
    inactive_namespace: str,
) -> None:
    class UnreadableConfig:
        def __deepcopy__(self, memo: dict[int, object]) -> UnreadableConfig:
            return self

        def __getattr__(self, name: str) -> object:
            raise AssertionError(f"inactive config was read: {name}")

    config = RuntimeConfig()
    config.control.active_algorithm = algorithm_id
    setattr(config.control.algorithms, inactive_namespace, UnreadableConfig())

    service = RuntimeService(
        config,
        models=SimpleNamespace(get_active_deployment=lambda: None),
        executors=ExecutorRegistry.from_config(config),
    )

    assert service.control_algorithms.active_controller.mode == algorithm_id


def test_algorithm_switch_resets_old_state_and_installs_a_fresh_active_controller() -> None:
    config = RuntimeConfig()
    service = RuntimeService(
        config,
        models=SimpleNamespace(get_active_deployment=lambda: None),
        executors=ExecutorRegistry.from_config(config),
    )
    previous = service.control_algorithms.active_controller
    reset_calls: list[str] = []
    previous_reset = previous.reset

    def record_reset() -> None:
        reset_calls.append("old")
        previous_reset()

    previous.reset = record_reset
    updated = copy.deepcopy(config)
    updated.control.active_algorithm = UNIVERSAL_SATURATED

    service.update_config(updated)

    assert reset_calls == ["old"]
    assert service.control_algorithms.active_algorithm_id == UNIVERSAL_SATURATED
    assert service.control_algorithms.active_controller is not previous
    assert service.control_algorithms.active_controller.state.residual_x_counts == 0.0
    assert service.control_algorithms.active_controller.state.residual_y_counts == 0.0


def test_runtime_schema_exposes_separate_algorithm_sections_and_product_labels() -> None:
    schema = runtime_config_schema()
    sections = {section["id"]: section for section in schema["sections"]}

    assert sections["control_shared"]["algorithm_scope"] == list(supported_algorithm_ids())
    shared_paths = {field["path"] for field in sections["control_shared"]["fields"]}
    assert "control.shared.trigger_activation_delay_ms" in shared_paths
    assert "control.aim.y_ratio" in shared_paths
    assert "control.configured_actuation_delay_s" not in shared_paths
    assert sections["control_universal_saturated"]["algorithm_scope"] == [
        UNIVERSAL_SATURATED
    ]
    assert sections["control_calibrated_angular"]["algorithm_scope"] == [CALIBRATED_ANGULAR]
    assert sections["control_robust_predictive"]["algorithm_scope"] == [
        DUAL_PHASE_ATAN_ROBUST_PREDICTIVE_V2
    ]

    selector = next(
        field
        for field in sections["control_shared"]["fields"]
        if field["path"] == "control.active_algorithm"
    )
    assert selector["option_labels"] == {
        UNIVERSAL_SATURATED: "通用控制",
        CALIBRATED_ANGULAR: "精确角度控制",
        DUAL_PHASE_ATAN_ROBUST_PREDICTIVE_V2: "稳健预测控制",
    }

    robust_paths = {
        field["path"] for field in sections["control_robust_predictive"]["fields"]
    }
    assert robust_paths
    assert all(
        path.startswith("control.algorithms.dual_phase_atan_robust_predictive_v2.")
        for path in robust_paths
    )
    standard_output_paths = {
        field["path"] for field in sections["control_standard_output"]["fields"]
    }
    assert "control.shared.recoil_enabled" in standard_output_paths
    assert "control.shared.trigger_activation_delay_ms" not in standard_output_paths
    non_predictive_paths = {
        field["path"]
        for field in sections["control_non_predictive_shared"]["fields"]
    }
    assert sections["control_non_predictive_shared"]["algorithm_scope"] == [
        UNIVERSAL_SATURATED,
        CALIBRATED_ANGULAR,
    ]
    assert non_predictive_paths == {"control.configured_actuation_delay_s"}
    algorithm_labels = {
        str(field["label"])
        for section_id in (
            "control_universal_saturated",
            "control_calibrated_angular",
            "control_robust_predictive",
        )
        for field in sections[section_id]["fields"]
    }
    assert all("通用适配" not in label for label in algorithm_labels)
    assert all("精确标定" not in label for label in algorithm_labels)
    assert all("精确 v2" not in label for label in algorithm_labels)


def test_studio_uses_product_names_for_three_exclusive_algorithm_pages() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "web"
        / "src"
        / "features"
        / "studio"
        / "StudioConsoleView.tsx"
    ).read_text(encoding="utf-8")

    assert 'label: "通用控制"' in source
    assert 'label: "精确角度控制"' in source
    assert 'label: "稳健预测控制"' in source
    assert 'aria-label="控制模式"' in source
    assert 'className="mini-segmented control-algorithm-segmented"' in source
    assert "精确双阶段稳健预测 v2" not in source
    assert "通用适配" not in source
    assert '<option value="calibrated_angular">精确标定</option>' not in source


def test_runtime_config_page_filters_sections_by_active_algorithm() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "web" / "src" / "features" / "config" / "ConfigView.tsx").read_text(
        encoding="utf-8"
    )
    api_source = (root / "web" / "src" / "api.ts").read_text(encoding="utf-8")

    assert "section.algorithm_scope.includes(activeAlgorithm)" in source
    assert "visibleSections.map((section)" in source
    assert "field.option_labels?.[option]" in source
    assert "algorithm_scope?: string[]" in api_source
    assert "option_labels?: Record<string, string>" in api_source


@pytest.mark.parametrize("algorithm_id", [UNIVERSAL_SATURATED, CALIBRATED_ANGULAR])
def test_non_predictive_control_uses_measured_aim_without_kalman_prediction(
    algorithm_id: str,
) -> None:
    config = RuntimeConfig()
    config.control.active_algorithm = algorithm_id
    config.control.scheduler_enabled = False
    executors = ExecutorRegistry.from_config(config)
    service = RuntimeService(
        config,
        models=SimpleNamespace(get_active_deployment=lambda: None),
        executors=executors,
    )
    service.running = True
    capture_ts_ns = time.monotonic_ns()
    batch = DetectionBatch(
        frame_id=1,
        generation=1,
        capture_ts_ns=capture_ts_ns,
        inference_start_ts_ns=capture_ts_ns + 1_000,
        inference_end_ts_ns=capture_ts_ns + 2_000,
        detections=[Detection(cls=0, score=0.95, x1=330, y1=250, x2=490, y2=568)],
        classes=["target"],
        coordinate_space="roi",
    )

    service.process_detection_batch(
        batch,
        width=640,
        height=640,
        source_width=1920,
        source_height=1080,
        roi_offset_x=600,
        roi_offset_y=220,
    )

    assert service.last_control is not None
    observation = service.last_control["mouse_observation"]
    assert observation["prediction_source"] == "none"
    assert observation["predicted_x_px"] == observation["observed_x_px"]
    assert observation["predicted_y_px"] == observation["observed_y_px"]
