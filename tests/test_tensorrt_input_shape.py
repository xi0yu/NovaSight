from __future__ import annotations

import pytest

from novasight.inference.tensorrt import _resolve_input_shape


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


def test_tensorrt_resolve_input_shape_uses_requested_dynamic_shape() -> None:
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

    assert shape == (1, 3, 320, 320)
    assert source == "requested"
    assert profile["opt"] == (1, 3, 640, 640)


def test_tensorrt_resolve_input_shape_rejects_static_shape_mismatch() -> None:
    with pytest.raises(RuntimeError, match="requested input shape"):
        _resolve_input_shape(
            object(),
            input_name="images",
            engine_shape=(1, 3, 640, 640),
            requested_shape=(1, 3, 320, 320),
        )

