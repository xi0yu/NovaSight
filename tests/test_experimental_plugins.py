import pytest

from novasight.plugins import Detection, FrameContext, PluginRuntime
from novasight.plugins.builtin import ExperimentalCenterControlPlugin


def _context(
    width: int = 1000,
    height: int = 500,
    detections: list[Detection] | None = None,
    classes: list[str] | None = None,
) -> FrameContext:
    return FrameContext(
        frame_id=1,
        width=width,
        height=height,
        detections=detections or [],
        classes=classes or ["target"],
    )


def _experimental_target_payload(context: FrameContext) -> dict:
    result = PluginRuntime.with_builtin_plugins().process(context)
    target = next(
        item
        for item in result.plugin_results
        if item.plugin_id == "vision.experimental_target"
    )
    return target.payload


def test_experimental_target_reports_normalized_offset() -> None:
    payload = _experimental_target_payload(
        _context(
            detections=[Detection(cls=0, score=0.8, x=600, y=200, w=100, h=100)],
        )
    )

    assert payload["class_name"] == "target"
    assert payload["center"] == {"x": 650.0, "y": 250.0}
    assert payload["normalized_offset"] == {"x": 0.3, "y": 0.0}


def test_experimental_center_control_outputs_pixel_delta() -> None:
    intent = ExperimentalCenterControlPlugin().process(
        _context(
            detections=[Detection(cls=0, score=0.8, x=600, y=200, w=100, h=100)],
        )
    )

    assert intent is not None
    assert intent.dx == 150
    assert intent.dy == 0
    assert intent.confidence == 0.8


def test_experimental_plugins_return_empty_target_without_detections() -> None:
    context = _context()

    assert _experimental_target_payload(context) == {"target": None}
    assert ExperimentalCenterControlPlugin().process(context) is None


def test_experimental_plugins_use_highest_score_detection() -> None:
    context = _context(
        detections=[
            Detection(cls=0, score=0.8, x=600, y=200, w=100, h=100),
            Detection(cls=1, score=0.9, x=400, y=100, w=50, h=50),
        ],
        classes=["target", "other"],
    )

    payload = _experimental_target_payload(context)
    intent = ExperimentalCenterControlPlugin().process(context)

    assert payload["class_name"] == "other"
    assert payload["score"] == 0.9
    assert payload["center"] == {"x": 425.0, "y": 125.0}
    assert payload["normalized_offset"] == {"x": -0.15, "y": 0.5}
    assert intent is not None
    assert intent.dx == -75
    assert intent.dy == 125
    assert intent.confidence == 0.9


def test_experimental_target_uses_string_for_out_of_range_class_index() -> None:
    payload = _experimental_target_payload(
        _context(
            detections=[Detection(cls=4, score=0.8, x=600, y=200, w=100, h=100)],
            classes=["target"],
        )
    )

    assert payload["class_name"] == "4"


@pytest.mark.parametrize(
    ("width", "height"),
    [
        (0, 500),
        (1000, 0),
        (-1, 500),
        (1000, -1),
    ],
)
def test_experimental_plugins_handle_invalid_frame_size(
    width: int,
    height: int,
) -> None:
    context = _context(
        width=width,
        height=height,
        detections=[Detection(cls=0, score=0.8, x=600, y=200, w=100, h=100)],
    )

    assert _experimental_target_payload(context) == {
        "target": None,
        "reason": "invalid frame size",
    }
    assert ExperimentalCenterControlPlugin().process(context) is None
