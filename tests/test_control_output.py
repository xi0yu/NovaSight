import pytest

from novasight.contracts import ControlIntent
from novasight.control import ControlOutputPolicy


def _intent(
    dx: float = 500, dy: float = -300, confidence: float = 0.9
) -> ControlIntent:
    return ControlIntent(
        dx=dx, dy=dy, action="move", confidence=confidence,
        reason="test", source_id="control.test",
    )


def test_policy_clamps_dx_dy() -> None:
    output = ControlOutputPolicy(max_abs_dx=120, max_abs_dy=80).apply(_intent())

    assert output.dx == 120
    assert output.dy == -80
    assert output.clipped is True
    assert "clamped" in output.reason


@pytest.mark.parametrize("kwargs", [{"max_abs_dx": -1}, {"max_abs_dy": -1}])
def test_policy_rejects_negative_bounds(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="max_abs_.*must be >= 0"):
        ControlOutputPolicy(**kwargs)


def test_policy_zero_bounds_clamp_to_zero_movement() -> None:
    output = ControlOutputPolicy(max_abs_dx=0, max_abs_dy=0).apply(_intent())

    assert output.accepted is True
    assert output.dx == 0
    assert output.dy == 0
    assert output.clipped is True
    assert "clamped" in output.reason


def test_policy_drops_low_confidence() -> None:
    output = ControlOutputPolicy(min_confidence=0.5).apply(_intent(confidence=0.1))

    assert output.accepted is False
    assert output.dx == 0
    assert output.dy == 0
    assert "confidence" in output.reason


def test_policy_rejects_non_finite_values() -> None:
    output = ControlOutputPolicy().apply(_intent(dx=float("nan")))

    assert output.accepted is False
    assert "non-finite" in output.reason


@pytest.mark.parametrize("confidence", [float("nan"), float("inf")])
def test_policy_rejects_non_finite_confidence(confidence: float) -> None:
    output = ControlOutputPolicy().apply(_intent(confidence=confidence))

    assert output.accepted is False
    assert output.dx == 0
    assert output.dy == 0
    assert "non-finite" in output.reason


def test_policy_preserves_scheduler_metadata() -> None:
    intent = ControlIntent(
        dx=10,
        dy=20,
        action="move",
        confidence=1.0,
        reason="test",
        source_id="control.test",
        source_frame_id=42,
        source_track_id=7,
        predicted_source=True,
    )

    output = ControlOutputPolicy().apply(intent)

    assert output.source_frame_id == 42
    assert output.source_track_id == 7
    assert output.predicted_source is True
