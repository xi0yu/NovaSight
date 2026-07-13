from __future__ import annotations

from dataclasses import asdict

import pytest

from novasight.config import parse_runtime_config
from novasight.executors.runtime import ExecutorRegistry, scheduler_from_config


def test_algorithm_configs_are_isolated_under_explicit_namespaces() -> None:
    config = parse_runtime_config(
        {
            "control": {
                "active_algorithm": "dual_phase_atan_predictive_v1",
                "algorithms": {
                    "calibrated_angular": {"kp_x": 0.8},
                    "dual_phase_atan_predictive_v1": {
                        "far": {"kp": 0.40},
                        "near": {"kp": 0.10},
                    },
                },
            }
        }
    )

    serialized = asdict(config)["control"]
    assert serialized["active_algorithm"] == "dual_phase_atan_predictive_v1"
    assert serialized["algorithms"]["calibrated_angular"]["kp_x"] == 0.8
    assert serialized["algorithms"]["dual_phase_atan_predictive_v1"]["far"]["kp"] == 0.40
    assert serialized["algorithms"]["dual_phase_atan_predictive_v1"]["near"]["kp"] == 0.10
    assert "calibrated_angular" not in serialized
    assert "dual_phase_atan_predictive_v1" not in serialized


def test_legacy_control_layout_migrates_without_changing_algorithm_values() -> None:
    config = parse_runtime_config(
        {
            "control": {
                "mode": "ttbox_pid_atan",
                "ttbox_pid_atan": {
                    "per_frame_gain_x": 0.17,
                    "per_frame_gain_y": 0.09,
                },
            }
        }
    )

    assert config.control.active_algorithm == "ttbox_pid_atan"
    assert config.control.algorithms.ttbox_pid_atan.per_frame_gain_x == 0.17
    assert config.control.algorithms.ttbox_pid_atan.per_frame_gain_y == 0.09


def test_dual_phase_v1_rejects_y_prediction_and_never_builds_scheduler() -> None:
    with pytest.raises(ValueError, match="prediction.enabled_y"):
        parse_runtime_config(
            {
                "control": {
                    "active_algorithm": "dual_phase_atan_predictive_v1",
                    "algorithms": {
                        "dual_phase_atan_predictive_v1": {"prediction": {"enabled_y": True}}
                    },
                }
            }
        )

    config = parse_runtime_config(
        {
            "control": {
                "active_algorithm": "dual_phase_atan_predictive_v1",
                "scheduler_enabled": True,
            }
        }
    )
    assert scheduler_from_config(config) is None


def test_robust_v2_namespace_uses_short_window_units_and_isolated_names() -> None:
    config = parse_runtime_config(
        {
            "control": {
                "active_algorithm": "dual_phase_atan_robust_predictive_v2",
                "algorithms": {
                    "dual_phase_atan_predictive_v1": {
                        "prediction": {"far_weight": 0.21},
                    },
                    "dual_phase_atan_robust_predictive_v2": {
                        "velocity": {
                            "smoothing_tau_ms": 25.0,
                            "spread_base_px_ms": 0.2,
                        },
                        "prediction": {"coefficient": 1.5},
                        "atan": {
                            "far": {"kp": 0.4},
                            "near": {"kp": 0.1},
                        },
                    },
                },
            }
        }
    )

    serialized = asdict(config)["control"]["algorithms"]
    robust = serialized["dual_phase_atan_robust_predictive_v2"]
    assert robust["velocity"]["history_size"] == 4
    assert robust["velocity"]["velocity_sample_count"] == 3
    assert robust["velocity"]["smoothing_tau_ms"] == 25.0
    assert robust["prediction"]["coefficient"] == 1.5
    assert robust["atan"]["far"]["kp"] == 0.4
    assert robust["atan"]["near"]["kp"] == 0.1
    assert serialized["dual_phase_atan_predictive_v1"]["prediction"]["far_weight"] == 0.21
    assert scheduler_from_config(config) is None
    executors = ExecutorRegistry.from_config(config)
    assert executors.single_command_per_observation is True
    assert executors.policy.max_abs_dx == 127
    assert executors.policy.max_abs_dy == 127


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"velocity": {"history_size": 3}}, "history_size=4"),
        ({"velocity": {"velocity_sample_count": 2}}, "velocity_sample_count=3"),
        ({"prediction": {"coefficient": 2.1}}, "coefficient"),
        ({"prediction": {"enabled_y": True}}, "enabled_y"),
        ({"atan": {"far": {"max_counts_per_update": 128}}}, "max_counts_per_update"),
    ],
)
def test_robust_v2_rejects_values_that_change_the_frozen_algorithm(
    payload: dict,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_runtime_config(
            {
                "control": {
                    "active_algorithm": "dual_phase_atan_robust_predictive_v2",
                    "algorithms": {
                        "dual_phase_atan_robust_predictive_v2": payload,
                    },
                }
            }
        )
