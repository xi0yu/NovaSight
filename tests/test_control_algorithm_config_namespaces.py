from __future__ import annotations

from dataclasses import asdict

import pytest

from novasight.config import parse_runtime_config
from novasight.executors.runtime import ExecutorRegistry, scheduler_from_config


def test_algorithm_configs_are_isolated_under_explicit_namespaces() -> None:
    config = parse_runtime_config(
        {
            "control": {
                "active_algorithm": "dual_phase_atan_robust_predictive_v2",
                "algorithms": {
                    "calibrated_angular": {"kp_x": 0.8},
                    "dual_phase_atan_robust_predictive_v2": {
                        "atan": {
                            "far": {"kp": 0.40},
                            "near": {"kp": 0.20},
                        },
                    },
                },
            }
        }
    )

    serialized = asdict(config)["control"]
    assert serialized["active_algorithm"] == "dual_phase_atan_robust_predictive_v2"
    assert serialized["algorithms"]["calibrated_angular"]["kp_x"] == 0.8
    robust = serialized["algorithms"]["dual_phase_atan_robust_predictive_v2"]
    assert robust["atan"]["far"]["kp"] == 0.40
    assert robust["atan"]["near"]["kp"] == 0.20
    assert "calibrated_angular" not in serialized
    assert "dual_phase_atan_robust_predictive_v2" not in serialized


@pytest.mark.parametrize(
    "removed_algorithm",
    ["ttbox_pid_atan", "dual_phase_atan_predictive_v1"],
)
def test_removed_active_algorithms_migrate_to_robust_v2(removed_algorithm: str) -> None:
    config = parse_runtime_config(
        {
            "control": {
                "active_algorithm": removed_algorithm,
                "algorithms": {removed_algorithm: {}},
            }
        }
    )

    serialized = asdict(config)["control"]
    assert serialized["active_algorithm"] == "dual_phase_atan_robust_predictive_v2"
    assert removed_algorithm not in serialized["algorithms"]


def test_robust_v2_never_builds_scheduler() -> None:
    config = parse_runtime_config(
        {
            "control": {
                "active_algorithm": "dual_phase_atan_robust_predictive_v2",
                "scheduler_enabled": True,
            }
        }
    )
    assert scheduler_from_config(config) is None


def test_robust_v2_namespace_uses_single_threshold_and_shared_atan_scale() -> None:
    config = parse_runtime_config(
        {
            "control": {
                "active_algorithm": "dual_phase_atan_robust_predictive_v2",
                "algorithms": {
                    "dual_phase_atan_predictive_v1": {
                        "prediction": {"far_weight": 0.21},
                    },
                    "dual_phase_atan_robust_predictive_v2": {
                        "mode": {"near_threshold_px": 14.0},
                        "velocity": {
                            "smoothing_tau_ms": 25.0,
                            "spread_base_px_ms": 0.2,
                        },
                        "prediction": {"coefficient": 1.5},
                        "atan": {
                            "scale_counts": 300.0,
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
    assert robust["mode"] == {"near_threshold_px": 14.0}
    assert robust["velocity"]["history_size"] == 4
    assert robust["velocity"]["velocity_sample_count"] == 3
    assert robust["velocity"]["smoothing_tau_ms"] == 25.0
    assert robust["prediction"]["coefficient"] == 1.5
    assert robust["atan"]["scale_counts"] == 300.0
    assert "scale_counts" not in robust["atan"]["far"]
    assert "scale_counts" not in robust["atan"]["near"]
    assert "dual_phase_atan_predictive_v1" not in serialized
    assert scheduler_from_config(config) is None
    executors = ExecutorRegistry.from_config(config)
    assert executors.single_command_per_observation is True
    assert executors.policy.max_abs_dx == 127
    assert executors.policy.max_abs_dy == 127


def test_v2_schema_two_migrates_to_single_threshold_and_shared_scale() -> None:
    config = parse_runtime_config(
        {
            "control": {
                "active_algorithm": "dual_phase_atan_robust_predictive_v2",
                "algorithms": {
                    "dual_phase_atan_robust_predictive_v2": {
                        "schema_version": 2,
                        "mode": {
                            "near_enter_min_px": 13.0,
                            "near_exit_min_px": 19.0,
                            "near_enter_bbox_h_ratio": 0.45,
                            "near_exit_bbox_h_ratio": 0.60,
                        },
                        "atan": {
                            "far": {"scale_counts": 280.0},
                            "near": {"scale_counts": 280.0},
                        },
                    }
                },
            }
        }
    )

    robust = config.control.dual_phase_atan_robust_predictive_v2
    assert robust.schema_version == 3
    assert robust.mode.near_threshold_px == 13.0
    assert robust.atan.scale_counts == 280.0


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
