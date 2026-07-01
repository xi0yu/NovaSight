import pytest

from novasight.control import ControlOutputPolicy
from novasight.plugins import ControlIntent


def _intent(dx=500, dy=-300, confidence=0.9):
    return ControlIntent(dx, dy, "move", confidence, "test", "control.test")


def test_policy_clamps_dx_dy() -> None:
    output = ControlOutputPolicy(max_abs_dx=120, max_abs_dy=80).apply(_intent())

    assert output.dx == 120
    assert output.dy == -80
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
