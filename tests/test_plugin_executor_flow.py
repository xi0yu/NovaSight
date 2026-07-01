from novasight.executors import ExecutorRegistry
from novasight.plugins import Detection, FrameContext, PluginRuntime, Track


def test_builtin_control_plugin_outputs_intent_and_dry_run_records_it() -> None:
    plugins = PluginRuntime.with_builtin_plugins()
    executors = ExecutorRegistry.with_builtin_executors(default="dry_run")
    context = FrameContext(
        frame_id=42,
        width=1280,
        height=720,
        detections=[Detection(cls=0, score=0.91, x=600, y=300, w=80, h=120)],
        tracks=[Track(track_id=7, cls=0, score=0.91, x=600, y=300, w=80, h=120)],
        classes=["target"],
    )

    results = plugins.process(context)
    intents = [
        item
        for item in results.control_intents
        if item.plugin_id == "control.center_target"
    ]
    assert len(intents) == 1
    assert intents[0].dx < 0
    assert intents[0].dy > 0

    execution = executors.execute(intents[0])
    assert execution.executor_id == "dry_run"
    assert execution.sent is False
    assert execution.intent == intents[0]


def test_kmnet_executor_is_unavailable_without_driver() -> None:
    executors = ExecutorRegistry.with_builtin_executors(default="kmnet")

    status = executors.status()

    assert status["selected"] == "kmnet"
    assert status["executors"]["kmnet"]["available"] is False
