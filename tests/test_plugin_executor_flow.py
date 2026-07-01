import builtins
import sys
from dataclasses import fields
from types import SimpleNamespace

from novasight.config import RuntimeConfig
from novasight.executors import ExecutorRegistry
from novasight.executors.dry_run import DryRunExecutor
from novasight.executors.kmnet import KmNetExecutor
from novasight.model_registry import ModelRegistry
from novasight.plugins import (
    ControlIntent,
    Detection,
    FrameContext,
    PluginBatchResult,
    PluginResult,
    PluginRuntime,
    Track,
)
from novasight.runtime import RuntimeService


def _context(
    detections: list[Detection] | None = None,
    tracks: list[Track] | None = None,
) -> FrameContext:
    return FrameContext(
        frame_id=42,
        width=1280,
        height=720,
        detections=detections or [],
        tracks=tracks or [],
        classes=["target"],
    )


def _intent(dx: float = -40.4, dy: float = 60.6) -> ControlIntent:
    return ControlIntent(
        dx=dx,
        dy=dy,
        action="move",
        confidence=0.91,
        reason="test intent",
        plugin_id="control.test",
    )


def _force_kmnet_import_failure(monkeypatch) -> None:
    original_import = builtins.__import__

    def fail_kmnet_import(name, *args, **kwargs):
        if name == "kmNet":
            raise RuntimeError("native load failed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_kmnet_import)
    monkeypatch.delitem(sys.modules, "kmNet", raising=False)


def test_contract_dataclasses_expose_planned_public_fields() -> None:
    assert [item.name for item in fields(PluginResult)] == [
        "plugin_id",
        "kind",
        "payload",
    ]
    assert [item.name for item in fields(ControlIntent)] == [
        "dx",
        "dy",
        "action",
        "confidence",
        "reason",
        "plugin_id",
    ]
    assert [item.name for item in fields(PluginBatchResult)] == [
        "plugin_results",
        "control_intents",
    ]


def test_builtin_plugin_metadata_is_introspectable_from_runtime() -> None:
    plugins = PluginRuntime.with_builtin_plugins()

    assert [
        (plugin.plugin_id, plugin.kind)
        for plugin in plugins.vision_plugins
    ] == [
        ("vision.track_stats", "vision"),
        ("vision.experimental_target", "vision"),
    ]
    assert [
        (plugin.plugin_id, plugin.kind)
        for plugin in plugins.control_plugins
    ] == [
        ("control.center_target", "control"),
        ("control.experimental_center", "control"),
    ]


def test_builtin_plugins_report_stats_and_center_intent_from_top_left_boxes() -> None:
    plugins = PluginRuntime.with_builtin_plugins()
    context = _context(
        detections=[Detection(cls=0, score=0.91, x=560, y=240, w=80, h=120)],
        tracks=[Track(track_id=7, cls=0, score=0.91, x=560, y=240, w=80, h=120)],
    )

    results = plugins.process(context)

    stats = [
        item
        for item in results.plugin_results
        if item.plugin_id == "vision.track_stats"
    ]
    assert len(stats) == 1
    assert stats[0].kind == "vision"
    assert stats[0].payload == {"detections": 1, "tracks": 1}

    intents = [
        item
        for item in results.control_intents
        if item.plugin_id == "control.center_target"
    ]
    assert len(intents) == 1
    assert intents[0].dx == -40
    assert intents[0].dy == 60
    assert intents[0].action == "move"
    assert intents[0].confidence == 0.91
    assert "track" in intents[0].reason


def test_center_target_prefers_highest_score_track_over_detection() -> None:
    plugins = PluginRuntime.with_builtin_plugins()
    context = _context(
        detections=[
            Detection(cls=0, score=0.99, x=1000, y=500, w=20, h=20),
        ],
        tracks=[
            Track(track_id=1, cls=0, score=0.10, x=100, y=100, w=20, h=20),
            Track(track_id=2, cls=0, score=0.80, x=700, y=300, w=40, h=40),
        ],
    )

    intent = plugins.process(context).control_intents[0]

    assert intent.dx == 80
    assert intent.dy == 40
    assert intent.confidence == 0.80
    assert "track 2" in intent.reason


def test_center_target_falls_back_to_highest_score_detection_without_tracks() -> None:
    plugins = PluginRuntime.with_builtin_plugins()
    context = _context(
        detections=[
            Detection(cls=0, score=0.10, x=100, y=100, w=20, h=20),
            Detection(cls=0, score=0.95, x=500, y=200, w=60, h=80),
        ],
    )

    intent = plugins.process(context).control_intents[0]

    assert intent.dx == -110
    assert intent.dy == 120
    assert intent.confidence == 0.95
    assert "detection" in intent.reason


def test_plugin_runtime_drops_none_control_intents() -> None:
    class NullControlPlugin:
        plugin_id = "control.none"

        def process(self, context: FrameContext) -> ControlIntent | None:
            return None

    runtime = PluginRuntime(control_plugins=[NullControlPlugin()])

    results = runtime.process(_context())

    assert results.control_intents == []


def test_runtime_service_process_frame_executes_plugin_control_intents(
    tmp_path,
) -> None:
    intent = _intent()

    class FakeControlPlugin:
        plugin_id = "control.fake"
        kind = "control"

        def process(self, context: FrameContext) -> ControlIntent:
            return intent

    dry_run = DryRunExecutor()
    service = RuntimeService(
        config=RuntimeConfig(),
        models=ModelRegistry(
            db_path=tmp_path / "novasight.db",
            data_dir=tmp_path / "models",
        ),
        plugins=PluginRuntime(control_plugins=[FakeControlPlugin()]),
        executors=ExecutorRegistry(executors=[dry_run], default="dry_run"),
    )

    result = service.process_frame(_context())

    assert result.plugin_batch.control_intents == [intent]
    assert [execution.intent for execution in result.execution_results] == [intent]
    assert dry_run.history == [intent]


def test_dry_run_records_intent_history() -> None:
    executor = DryRunExecutor()
    intent = _intent()

    execution = executor.execute(intent)

    assert execution.executor_id == "dry_run"
    assert execution.sent is False
    assert execution.intent == intent
    assert executor.history == [intent]
    assert executor.available() is True


def test_registry_status_uses_callable_availability(monkeypatch) -> None:
    _force_kmnet_import_failure(monkeypatch)
    executors = ExecutorRegistry.with_builtin_executors(default="kmnet")

    status = executors.status()

    assert status["selected"] == "kmnet"
    assert status["executors"]["dry_run"]["available"] is True
    assert status["executors"]["kmnet"]["available"] is False


def test_kmnet_executor_unavailable_execute_returns_unsent(monkeypatch) -> None:
    _force_kmnet_import_failure(monkeypatch)
    executor = KmNetExecutor()

    execution = executor.execute(_intent())

    assert executor.available() is False
    assert execution.executor_id == "kmnet"
    assert execution.sent is False
    assert execution.message == "kmNet driver unavailable"


def test_kmnet_executor_sends_rounded_move_to_available_driver(monkeypatch) -> None:
    moves: list[tuple[int, int]] = []
    fake_kmnet = SimpleNamespace(move=lambda dx, dy: moves.append((dx, dy)))
    monkeypatch.setitem(sys.modules, "kmNet", fake_kmnet)
    executor = KmNetExecutor()

    execution = executor.execute(_intent(dx=10.4, dy=-20.6))

    assert executor.available() is True
    assert moves == [(10, 21)]
    assert execution.sent is True
