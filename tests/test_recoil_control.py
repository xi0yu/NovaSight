from __future__ import annotations

from dataclasses import replace
import time

import pytest

from novasight.config import RuntimeConfig
from novasight.contracts import ControlIntent
from novasight.control import ControlMixer
from novasight.control.recoil import (
    RecoilDecision,
    RecoilInput,
    RecoilState,
    TargetRelativeRecoilConfig,
    TargetRelativeRecoilController,
)
from novasight.executors import ExecutionResult, ExecutorRegistry


def _decision(
    state: RecoilState,
    *,
    emitted: int = 0,
    gate: float = 1.0,
    base: float = 600.0,
) -> RecoilDecision:
    return RecoilDecision(
        state=state,
        base_rate_counts_s=base,
        fast_add_rate_counts_s=0.0,
        gate=gate,
        final_rate_counts_s=base * gate,
        requested_counts_y=float(emitted),
        emitted_counts_y=emitted,
        residual_counts_y=0.0,
        block_reason="POSITION_BRAKE" if gate == 0.0 else "",
    )


def test_recoil_controller_complete_rate_state_and_freshness_contract() -> None:
    config = TargetRelativeRecoilConfig(
        enabled=True,
        base_rate_counts_s=600.0,
        max_rate_counts_s=1000.0,
        startup_ms=0.0,
        fast_add_gain_counts_s=500.0,
        stale_threshold_ms=55.0,
    )

    def one_second(hz: int) -> float:
        controller = TargetRelativeRecoilController(config)
        now_ns = 1_000_000_000
        emitted = 0
        decision = None
        for _ in range(hz):
            now_ns += round(1_000_000_000 / hz)
            decision = controller.calculate(
                RecoilInput(True, now_ns, 1.0 / hz, True, 7, 1, 4.0, 0.0)
            )
            emitted += decision.emitted_counts_y
        assert decision is not None
        return emitted + decision.residual_counts_y

    assert one_second(60) == pytest.approx(600.0, abs=1e-6)
    assert one_second(120) == pytest.approx(600.0, abs=1e-6)

    controller = TargetRelativeRecoilController(config)
    active = controller.calculate(RecoilInput(True, 1_000_000_000, 0.004, True, 1, 1, 4.0, 0.20))
    brake = controller.calculate(RecoilInput(True, 1_004_000_000, 0.004, True, 1, 2, 4.0, -0.08))
    stopped = controller.calculate(RecoilInput(True, 1_008_000_000, 0.004, True, 1, 3, 4.0, -0.12))
    stale = controller.calculate(RecoilInput(True, 1_012_000_000, 0.004, True, 1, 4, 60.0, 0.0))
    released = controller.calculate(RecoilInput(False, 1_016_000_000, 0.004, True, 1, 5, 4.0, 0.0))

    assert active.state is RecoilState.ACTIVE
    assert active.fast_add_rate_counts_s > 0.0
    assert brake.state is RecoilState.BRAKE and 0.0 < brake.gate < 1.0
    assert stopped.state is RecoilState.BRAKE and stopped.final_rate_counts_s == 0.0
    assert stale.state is RecoilState.STALE and stale.emitted_counts_y == 0
    assert released.state is RecoilState.IDLE and released.residual_counts_y == 0.0


@pytest.mark.parametrize(
    ("state", "gate", "tracking_y", "recoil_y", "expected_y"),
    [
        (RecoilState.IDLE, 0.0, 4.0, 0, 4.0),
        (RecoilState.ACTIVE, 1.0, 4.0, 2, 2.0),
        (RecoilState.ACTIVE, 1.0, -3.0, 2, -1.0),
        (RecoilState.BRAKE, 0.0, 4.0, 0, 0.0),
        (RecoilState.BRAKE, 0.0, -3.0, 0, -3.0),
        (RecoilState.STALE, 0.0, -3.0, 0, -3.0),
    ],
)
def test_recoil_mixer_has_one_vertical_owner_without_blocking_upward_recovery(
    state: RecoilState,
    gate: float,
    tracking_y: float,
    recoil_y: int,
    expected_y: float,
) -> None:
    mixed = ControlMixer().mix(5.0, tracking_y, _decision(state, emitted=recoil_y, gate=gate))

    assert mixed.final_x == 5.0
    assert mixed.final_y == expected_y


class _RecordingKmNet:
    executor_id = "kmnet"

    def __init__(self) -> None:
        self.left = True
        self.outputs = []

    def available(self) -> bool:
        return True

    def read_buttons(self) -> dict[str, object]:
        return {
            "available": True,
            "left": self.left,
            "right": not self.left,
            "side": False,
        }

    def execute(self, output):
        self.outputs.append(output)
        return ExecutionResult(self.executor_id, True, output, "sent")


def test_recoil_latest_replace_sends_once_per_tick_and_requires_live_left_button() -> None:
    config = RuntimeConfig()
    config.control.recoil.enabled = True
    config.control.recoil.base_rate_counts_s = 600.0
    config.control.recoil.max_rate_counts_s = 1000.0
    registry = ExecutorRegistry.from_config(config)
    device = _RecordingKmNet()
    registry.executors["kmnet"] = device
    expires_ns = time.monotonic_ns() + 1_000_000_000
    source = ControlIntent(
        dx=5.0,
        dy=6.0,
        action="move",
        confidence=1.0,
        reason="tracking",
        source_id="test",
        source_frame_id=7,
        source_track_id=3,
        trajectory_generation=7,
        trigger_required=False,
        trigger_active=True,
        command_expires_ts_ns=expires_ns,
    )
    registry.execute(source)
    recoil = _decision(RecoilState.ACTIVE, emitted=1)
    now_s = time.monotonic() + 0.005

    first = registry.tick_pending(now_s=now_s, recoil=recoil, recoil_source=source)
    second = registry.tick_pending(now_s=now_s + 0.005, recoil=recoil, recoil_source=source)
    newer_source = replace(
        source,
        source_frame_id=8,
        trajectory_generation=8,
        dx=4.0,
        dy=5.0,
    )
    registry.execute(newer_source)
    generation_mismatch = registry.tick_pending(
        now_s=now_s + 0.010,
        recoil=recoil,
        recoil_source=source,
    )
    device.left = False
    blocked = registry.tick_pending(now_s=now_s + 0.015, recoil=recoil, recoil_source=source)

    assert first.sent is True and second.sent is True and generation_mismatch.sent is True
    assert [(item.dx, item.dy) for item in device.outputs] == [(5, 1), (0, 1), (4, 5)]
    assert generation_mismatch.metadata["scheduler"]["recoil_generation_mismatch"] is True
    assert device.outputs[0].source_generation == device.outputs[1].source_generation == 7
    assert device.outputs[0].actuation_sequence < device.outputs[1].actuation_sequence
    assert blocked.sent is False
    assert blocked.metadata["block_reason"] == "TRIGGER_INACTIVE"
