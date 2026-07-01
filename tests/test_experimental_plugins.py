from novasight.plugins import Detection, FrameContext, PluginRuntime


def test_experimental_target_reports_normalized_offset() -> None:
    runtime = PluginRuntime.with_builtin_plugins()
    context = FrameContext(
        frame_id=1,
        width=1000,
        height=500,
        detections=[Detection(cls=0, score=0.8, x=600, y=200, w=100, h=100)],
        classes=["target"],
    )

    result = runtime.process(context)
    target = next(
        item
        for item in result.plugin_results
        if item.plugin_id == "vision.experimental_target"
    )

    assert target.payload["class_name"] == "target"
    assert target.payload["center"] == {"x": 650.0, "y": 250.0}
    assert target.payload["normalized_offset"] == {"x": 0.3, "y": 0.0}


def test_experimental_center_control_outputs_pixel_delta() -> None:
    runtime = PluginRuntime.with_builtin_plugins()
    context = FrameContext(
        frame_id=1,
        width=1000,
        height=500,
        detections=[Detection(cls=0, score=0.8, x=600, y=200, w=100, h=100)],
        classes=["target"],
    )

    intent = next(
        item
        for item in runtime.process(context).control_intents
        if item.plugin_id == "control.experimental_center"
    )

    assert intent.dx == 150
    assert intent.dy == 0
    assert intent.confidence == 0.8
