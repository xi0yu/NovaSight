from __future__ import annotations

from types import SimpleNamespace

from novasight.capture.preview import draw_overlay
from novasight.plugins import Detection


def test_preview_overlay_uses_original_pixel_coordinates(monkeypatch) -> None:
    calls: dict[str, list[tuple]] = {"rectangles": [], "lines": [], "circles": []}
    fake_cv2 = SimpleNamespace(
        FONT_HERSHEY_SIMPLEX=0,
        LINE_AA=16,
        circle=lambda image, center, radius, color, thickness: calls["circles"].append(
            (center, radius)
        ),
        rectangle=lambda image, p1, p2, color, thickness: calls["rectangles"].append(
            (p1, p2)
        ),
        line=lambda image, p1, p2, color, thickness: calls["lines"].append((p1, p2)),
        putText=lambda *args, **kwargs: None,
    )
    monkeypatch.setitem(__import__("sys").modules, "cv2", fake_cv2)

    draw_overlay(
        object(),
        width=1920,
        height=1080,
        detections=[Detection(cls=0, score=0.9, x=100, y=120, w=80, h=40)],
    )

    assert calls["circles"][0][0] == (960, 540)
    assert calls["rectangles"] == [((100, 120), (180, 160))]
    assert calls["lines"] == [((960, 540), (140, 140))]
