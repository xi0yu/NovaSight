from __future__ import annotations

from novasight.control.humanized_motion import HumanizedMotionGenerator, HumanizedMotionInput
from novasight.motion_profile import MotionProfileRepository
from novasight.config import RuntimeConfig
from novasight.executors import ExecutorRegistry
from novasight.runtime.service import RuntimeService
from types import SimpleNamespace


def _input(*, hold_ms: float, target_id: int = 1, active: bool = True) -> HumanizedMotionInput:
    return HumanizedMotionInput(
        base_x=12.0,
        base_y=4.0,
        full_x=120.0,
        full_y=40.0,
        error_x_px=240.0,
        error_y_px=80.0,
        target_width_px=32.0,
        target_id=target_id,
        trigger_active=active,
        left_trigger_active=active,
        trigger_hold_ms=hold_ms,
    )


def test_disabled_humanized_motion_is_exact_identity() -> None:
    result = HumanizedMotionGenerator(None).apply(_input(hold_ms=20.0))
    assert (result.x, result.y) == (12.0, 4.0)
    assert result.telemetry["humanized_motion_enabled"] is False


def test_profile_curve_generates_progressive_counts_and_resets_on_target_switch() -> None:
    profile = {
        "timing": {"fitts_a_ms": 40.0, "fitts_b_ms": 40.0},
        "progress_curve": [0.0, 0.1, 0.35, 0.7, 0.92, 1.0],
        "runtime_parameters": {"correction_start_ratio": 0.8, "correction_gain": 0.3},
    }
    generator = HumanizedMotionGenerator(profile)
    first = generator.apply(_input(hold_ms=10.0))
    second = generator.apply(_input(hold_ms=30.0))
    switched = generator.apply(_input(hold_ms=40.0, target_id=2))
    assert first.x == 0.0
    assert second.x > 0.0
    assert second.telemetry["humanized_motion_progress"] > 0.0
    assert switched.telemetry["humanized_motion_rebased"] is True
    assert switched.x == 0.0


def test_training_derives_fitts_and_progress_curve_from_samples(tmp_path) -> None:
    repository = MotionProfileRepository(tmp_path)
    session = repository.create_session("test")
    for index, target_x in enumerate((120.0, 240.0, 420.0, 620.0)):
        duration_us = 120_000 + index * 70_000
        repository.add_sample(session["session_id"], {
            "target_x": target_x,
            "target_y": 100.0,
            "radius_px": 20.0,
            "quality": "valid",
            "points": [
                {"t_us": 0, "x": 20.0, "y": 100.0},
                {"t_us": duration_us * 0.25, "x": 20.0 + (target_x - 20.0) * 0.12, "y": 100.0},
                {"t_us": duration_us * 0.55, "x": 20.0 + (target_x - 20.0) * 0.62, "y": 100.0},
                {"t_us": duration_us, "x": target_x, "y": 100.0},
            ],
        })
    profile = repository.train_profile(session["session_id"], "trained")
    assert profile["profile_version"] == 2
    assert profile["timing"]["model"] == "fitts"
    assert profile["timing"]["fitts_b_ms"] > 15.0
    assert len(profile["progress_curve"]) == 16
    assert profile["progress_curve"][0] == 0.0
    assert profile["progress_curve"][-1] == 1.0
    assert profile["runtime_parameters"]["correction_start_ratio"] != 0.68


def test_runtime_profile_activation_does_not_mutate_static_config(tmp_path) -> None:
    config = RuntimeConfig()
    repository = MotionProfileRepository(tmp_path)
    service = RuntimeService(
        config,
        models=SimpleNamespace(get_active_deployment=lambda: None),
        executors=ExecutorRegistry.from_config(config),
        motion_profile_repository=repository,
    )
    profile = {
        "profile_id": "runtime-only",
        "name": "runtime",
        "sample_count": 30,
        "progress_curve": [0.0, 0.3, 0.8, 1.0],
    }
    status = service.set_humanized_motion_profile(profile)
    assert status["enabled"] is True
    assert status["source"] == "runtime_memory"
    assert config.control.humanized_motion.enabled is False
    assert config.control.humanized_motion.active_profile == ""
    disabled = service.set_humanized_motion_profile(None)
    assert disabled["enabled"] is False
