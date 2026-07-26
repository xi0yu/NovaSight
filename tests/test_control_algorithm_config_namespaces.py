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


def test_robust_v2_builds_mandatory_latest_replace_scheduler() -> None:
    config = parse_runtime_config(
        {
            "control": {
                "active_algorithm": "dual_phase_atan_robust_predictive_v2",
                "scheduler_enabled": True,
            }
        }
    )
    assert scheduler_from_config(config) is not None


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
                            "smoothing_frames": 3.0,
                            "spread_base_px_ms": 0.2,
                        },
                        "prediction": {"lead_frames": 1.5},
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
    assert robust["velocity"]["smoothing_frames"] == 3.0
    assert robust["prediction"]["lead_frames"] == 1.5
    assert robust["atan"]["scale_counts"] == 300.0
    assert "scale_counts" not in robust["atan"]["far"]
    assert "scale_counts" not in robust["atan"]["near"]
    assert "dual_phase_atan_predictive_v1" not in serialized
    assert scheduler_from_config(config) is not None
    executors = ExecutorRegistry.from_config(config)
    assert executors.single_command_per_observation is False
    assert executors.latest_replace is True
    assert executors.scheduler is not None
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
    assert robust.schema_version == 8
    assert robust.mode.near_threshold_px == 13.0
    assert robust.atan.scale_counts == 280.0


def test_v2_schema_three_time_prediction_migrates_to_frame_units() -> None:
    config = parse_runtime_config(
        {
            "control": {
                "algorithms": {
                    "dual_phase_atan_robust_predictive_v2": {
                        "schema_version": 3,
                        "velocity": {"smoothing_tau_ms": 25.0},
                        "prediction": {
                            "coefficient": 1.5,
                            "actuation_delay_ms": 5.0,
                            "max_horizon_ms": 35.0,
                        },
                    }
                }
            }
        }
    )

    robust = config.control.dual_phase_atan_robust_predictive_v2
    assert robust.schema_version == 8
    assert robust.velocity.smoothing_frames == pytest.approx(3.0)
    assert robust.prediction.lead_frames == pytest.approx(1.5)


def test_v2_schema_seven_repairs_only_the_aggressive_generated_profile() -> None:
    migrated = parse_runtime_config(
        {
            "control": {
                "algorithms": {
                    "dual_phase_atan_robust_predictive_v2": {
                        "schema_version": 7,
                        "atan": {
                            "scale_counts": 1024.0,
                            "far": {"kp": 0.90, "max_counts_per_update": 600.0},
                            "near": {"kp": 0.30, "max_counts_per_update": 120.0},
                        },
                    }
                }
            }
        }
    ).control.dual_phase_atan_robust_predictive_v2
    customized = parse_runtime_config(
        {
            "control": {
                "algorithms": {
                    "dual_phase_atan_robust_predictive_v2": {
                        "schema_version": 7,
                        "atan": {
                            "scale_counts": 480.0,
                            "far": {"kp": 0.70, "max_counts_per_update": 420.0},
                            "near": {"kp": 0.25, "max_counts_per_update": 90.0},
                        },
                    }
                }
            }
        }
    ).control.dual_phase_atan_robust_predictive_v2

    assert migrated.schema_version == 8
    assert migrated.atan.scale_counts == 256.0
    assert migrated.atan.far.kp == 0.45
    assert migrated.atan.far.max_counts_per_update == 127.0
    assert migrated.atan.near.kp == 0.22
    assert migrated.atan.near.max_counts_per_update == 72.0
    assert customized.schema_version == 8
    assert customized.atan.scale_counts == 480.0
    assert customized.atan.far.kp == 0.70
    assert customized.atan.far.max_counts_per_update == 420.0
    assert customized.atan.near.kp == 0.25
    assert customized.atan.near.max_counts_per_update == 90.0


def test_v2_migration_preserves_explicitly_disabled_prediction() -> None:
    config = parse_runtime_config(
        {
            "control": {
                "algorithms": {
                    "dual_phase_atan_robust_predictive_v2": {
                        "schema_version": 3,
                        "prediction": {"enabled_x": False, "coefficient": 1.5},
                    }
                }
            }
        }
    )

    prediction = config.control.dual_phase_atan_robust_predictive_v2.prediction
    assert prediction.enabled is False
    assert prediction.lead_frames == 1.0


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"velocity": {"history_size": 3}}, "history_size=4"),
        ({"velocity": {"velocity_sample_count": 2}}, "velocity_sample_count=3"),
        ({"prediction": {"lead_frames": 10.1}}, "lead_frames"),
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
