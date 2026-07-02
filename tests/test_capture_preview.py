from __future__ import annotations

from novasight.capture.preview import draw_overlay
from novasight.plugins import Detection


def test_preview_overlay_renders_without_opencv() -> None:
    from PIL import Image

    output = draw_overlay(
        Image.new("RGB", (1920, 1080), (0, 0, 0)),
        width=1920,
        height=1080,
        detections=[Detection(cls=0, score=0.9, x=100, y=120, w=80, h=40)],
    )

    assert output.size == (1920, 1080)
    assert output.getpixel((100, 120)) == (80, 190, 255)
