from __future__ import annotations

from dataclasses import asdict

import pytest

from novasight.config import parse_runtime_config
from novasight.executors.runtime import scheduler_from_config


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
