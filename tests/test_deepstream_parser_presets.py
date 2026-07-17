from __future__ import annotations

import pytest

from novasight.deepstream.parser_presets import (
    normalize_parser_preset,
    parser_preset_payload,
    resolve_efficient_nms_parser_plan,
    resolve_parser_plan,
)


def test_auto_preset_resolves_v5_objectness_contract() -> None:
    plan = resolve_parser_plan(
        "auto",
        output_shape=[1, 7, 6300],
        class_count=2,
        inferred_has_objectness=True,
    )

    assert plan.compatibility == "yolov5"
    assert plan.has_objectness is True
    assert plan.parser_function == "NvDsInferParseNovaSight"


@pytest.mark.parametrize("preset", ["yolov8", "yolo11", "novasight_generic"])
def test_v8_v11_and_generic_use_the_builtin_raw_parser(preset: str) -> None:
    plan = resolve_parser_plan(
        preset,
        output_shape=[1, 8400, 6],
        class_count=2,
        inferred_has_objectness=False,
    )

    assert plan.compatibility == "yolov8_yolo11"
    assert plan.parser_library == "novasight_builtin"
    assert plan.nms_owner == "deepstream"


def test_explicit_v5_rejects_detected_v8_contract() -> None:
    with pytest.raises(ValueError, match="conflicts with the detected objectness"):
        resolve_parser_plan(
            "yolov5",
            output_shape=[1, 6, 8400],
            class_count=2,
            inferred_has_objectness=False,
        )


def test_custom_alias_means_novasight_builtin_generic_parser() -> None:
    assert normalize_parser_preset("custom") == "novasight_generic"
    payload = parser_preset_payload()

    assert all(item["external_library"] is False for item in payload)
    assert any(item["id"] == "novasight_generic" for item in payload)


def test_efficient_nms_plan_keeps_nms_owned_by_model() -> None:
    plan = resolve_efficient_nms_parser_plan("auto")

    assert plan.compatibility == "efficientnms"
    assert plan.nms_owner == "model"
    assert plan.parser_function == "NvDsInferParseNovaSight"
