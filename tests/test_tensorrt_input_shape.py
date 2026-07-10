from __future__ import annotations

from novasight.inference.tensorrt import (
    _resolve_input_shape,
    _select_detection_output_name,
)


class _ProfiledEngine:
    def __init__(
        self,
        *,
        minimum: tuple[int, int, int, int],
        optimum: tuple[int, int, int, int],
        maximum: tuple[int, int, int, int],
    ) -> None:
        self._profile = (minimum, optimum, maximum)

    def get_tensor_profile_shape(self, input_name: str, profile_index: int):
        assert input_name == "images"
        assert profile_index == 0
        return self._profile


def test_tensorrt_resolve_input_shape_uses_engine_profile_opt_over_requested_dynamic_shape() -> None:
    shape, source, profile = _resolve_input_shape(
        _ProfiledEngine(
            minimum=(1, 3, 256, 256),
            optimum=(1, 3, 640, 640),
            maximum=(1, 3, 1280, 1280),
        ),
        input_name="images",
        engine_shape=(-1, 3, -1, -1),
        requested_shape=(1, 3, 320, 320),
    )

    assert shape == (1, 3, 640, 640)
    assert source == "engine_profile_opt"
    assert profile["opt"] == (1, 3, 640, 640)


def test_tensorrt_resolve_input_shape_uses_static_engine_shape_over_requested_mismatch() -> None:
    shape, source, profile = _resolve_input_shape(
        object(),
        input_name="images",
        engine_shape=(1, 3, 640, 640),
        requested_shape=(1, 3, 320, 320),
    )

    assert shape == (1, 3, 640, 640)
    assert source == "engine_static"
    assert profile == {}


def test_tensorrt_selects_decodable_detection_output_instead_of_first_3d_tensor() -> None:
    selected, columns = _select_detection_output_name(
        ["boxes", "output0"],
        {
            "boxes": (1, 300, 4),
            "output0": (1, 84, 8400),
        },
        class_count=80,
    )

    assert selected == "output0"
    assert columns == 84


def test_tensorrt_rejects_outputs_that_current_decoder_cannot_consume() -> None:
    selected, columns = _select_detection_output_name(
        ["boxes", "scores", "classes"],
        {
            "boxes": (1, 300, 4),
            "scores": (1, 300),
            "classes": (1, 300),
        },
        class_count=80,
    )

    assert selected is None
    assert columns is None
