from __future__ import annotations

import json
from types import SimpleNamespace

from novasight.main import _doctor_gst_cpu_latest_smoke_report, main


def _metric_sample(elapsed_s: float) -> dict[str, object]:
    return {
        "elapsed_s": elapsed_s,
        "memory_rss_mb": 316.0,
        "capture_fps": 119.4,
        "published_frames": 7164,
        "overwritten_frames": 420,
        "acquired_frames": 6744,
        "latest_frame_broker_max_pending_depth": 1,
        "latest_frame_age_ms": 4.5,
        "preprocess_ms": 1.3,
        "h2d_ms": 0.4,
        "host_frame_copy_ms": 0.2,
        "inference_ms": 5.2,
        "postprocess_ms": 0.1,
        "batch_age_ms": 11.8,
        "stale_drop_count": 3,
        "control_observe_fps": 58.0,
        "appsink_caps": "video/x-raw,format=BGRx,width=640,height=640",
        "actual_pipeline_string": (
            "v4l2src device=/dev/video0 io-mode=2 do-timestamp=true ! "
            "queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream ! "
            "jpegparse ! nvv4l2decoder mjpeg=1 ! nvvidconv left=640 right=1280 top=220 bottom=860 ! "
            "video/x-raw,format=BGRx,width=640,height=640 ! "
            "queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream ! "
            "appsink name=sink emit-signals=false max-buffers=1 drop=true sync=false"
        ),
        "inference_selected": "tensorrt",
        "inference_device": "cuda",
        "preprocess_backend": "cpu",
    }


def _complete_report() -> dict[str, object]:
    return {
        "schema_version": 1,
        "check": "gst-cpu-latest-smoke",
        "accepted": True,
        "reason": "",
        "exit_code": 0,
        "parameters": {
            "device": "/dev/video0",
            "seconds": 60.0,
            "max_memory_growth_mb": 64.0,
        },
        "evidence": {
            "capture_backend": "gst_cpu_latest:nvmm-mjpg-iomode2",
            "capture_memory": "system",
            "capture_fps": 119.4,
            "published_frames": 7164,
            "overwritten_frames": 420,
            "acquired_frames": 6744,
            "latest_frame_broker_max_pending_depth": 1,
            "latest_frame_age_ms": 4.5,
            "appsink_caps": "video/x-raw,format=BGRx,width=640,height=640",
            "actual_pipeline_string": (
                "v4l2src device=/dev/video0 io-mode=2 do-timestamp=true ! "
                "queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream ! "
                "jpegparse ! nvv4l2decoder mjpeg=1 ! nvvidconv left=640 right=1280 top=220 bottom=860 ! "
                "video/x-raw,format=BGRx,width=640,height=640 ! "
                "queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream ! "
                "appsink name=sink emit-signals=false max-buffers=1 drop=true sync=false"
            ),
            "inference_selected": "tensorrt",
            "inference_device": "cuda",
            "inference_require_gpu": True,
            "inference_allow_cpu_fallback": False,
            "preprocess_backend": "cpu",
            "preprocess_ms": 1.3,
            "h2d_ms": 0.4,
            "host_frame_copy_ms": 0.2,
            "inference_ms": 5.2,
            "postprocess_ms": 0.1,
            "batch_age_ms": 11.8,
            "stale_drop_count": 3,
            "control_observe_fps": 58.0,
            "freshness_gate_rejected": False,
            "duration_s": 60.2,
            "memory_rss_start_mb": 312.0,
            "memory_rss_end_mb": 328.0,
            "memory_growth_mb": 16.0,
            "detection_batch": {
                "frame_id": 7164,
                "capture_ts_ns": 1_000_000,
                "inference_start_ts_ns": 1_004_000,
                "inference_end_ts_ns": 1_009_000,
                "control_now_ts_ns": 1_020_000,
                "model_input_size": [640, 640],
                "coordinate_space": "roi",
            },
        },
        "metric_samples": [
            *[_metric_sample(float(second)) for second in range(0, 61)],
            _metric_sample(60.2),
        ],
    }


def test_gst_cpu_latest_smoke_report_accepts_complete_runtime_evidence(tmp_path, capsys) -> None:
    report_path = tmp_path / "gst-cpu-latest-smoke.json"
    report_path.write_text(json.dumps(_complete_report()), encoding="utf-8")

    result = _doctor_gst_cpu_latest_smoke_report(
        SimpleNamespace(report_json=str(report_path))
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "accepted: True" in output
    assert "capture_backend: gst_cpu_latest:nvmm-mjpg-iomode2" in output
    assert "inference_selected: tensorrt" in output
    assert "control_observe_fps: 58.00" in output


def test_gst_cpu_latest_smoke_report_rejects_missing_control_observation(tmp_path, capsys) -> None:
    report = _complete_report()
    evidence = report["evidence"]
    assert isinstance(evidence, dict)
    evidence["control_observe_fps"] = 0.0
    evidence["freshness_gate_rejected"] = False
    report_path = tmp_path / "gst-cpu-latest-smoke.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    result = _doctor_gst_cpu_latest_smoke_report(
        SimpleNamespace(report_json=str(report_path))
    )

    output = capsys.readouterr().out
    assert result == 2
    assert "reason: gst_cpu_latest_smoke_report_invalid" in output
    assert "missing: evidence.control_observe_fps>0 unless freshness_gate_rejected" in output


def test_gst_cpu_latest_smoke_report_rejects_missing_copy_cost(tmp_path, capsys) -> None:
    report = _complete_report()
    evidence = report["evidence"]
    assert isinstance(evidence, dict)
    evidence.pop("host_frame_copy_ms")
    report_path = tmp_path / "gst-cpu-latest-smoke.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    result = _doctor_gst_cpu_latest_smoke_report(
        SimpleNamespace(report_json=str(report_path))
    )

    output = capsys.readouterr().out
    assert result == 2
    assert "missing: evidence.host_frame_copy_ms is required" in output


def test_gst_cpu_latest_smoke_report_rejects_missing_control_timestamp(tmp_path, capsys) -> None:
    report = _complete_report()
    evidence = report["evidence"]
    assert isinstance(evidence, dict)
    detection_batch = evidence["detection_batch"]
    assert isinstance(detection_batch, dict)
    detection_batch.pop("control_now_ts_ns")
    report_path = tmp_path / "gst-cpu-latest-smoke.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    result = _doctor_gst_cpu_latest_smoke_report(
        SimpleNamespace(report_json=str(report_path))
    )

    output = capsys.readouterr().out
    assert result == 2
    assert "missing: evidence.detection_batch.control_now_ts_ns is required" in output


def test_gst_cpu_latest_smoke_report_rejects_cpu_fallback_or_fifo_pipeline(tmp_path, capsys) -> None:
    report = _complete_report()
    evidence = report["evidence"]
    assert isinstance(evidence, dict)
    evidence["inference_allow_cpu_fallback"] = True
    evidence["actual_pipeline_string"] = "v4l2src ! queue ! appsink"
    report_path = tmp_path / "gst-cpu-latest-smoke.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    result = _doctor_gst_cpu_latest_smoke_report(
        SimpleNamespace(report_json=str(report_path))
    )

    output = capsys.readouterr().out
    assert result == 2
    assert "missing: evidence.inference_allow_cpu_fallback==false" in output
    assert "missing: evidence.actual_pipeline_string contains latest-only queue" in output
    assert "missing: evidence.actual_pipeline_string contains latest-only appsink" in output


def test_gst_cpu_latest_smoke_report_rejects_any_non_latest_queue(tmp_path, capsys) -> None:
    report = _complete_report()
    evidence = report["evidence"]
    assert isinstance(evidence, dict)
    evidence["actual_pipeline_string"] = (
        "v4l2src device=/dev/video0 io-mode=2 do-timestamp=true ! "
        "queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream ! "
        "jpegparse ! nvv4l2decoder mjpeg=1 ! nvvidconv left=640 right=1280 top=220 bottom=860 ! "
        "video/x-raw,format=BGRx,width=640,height=640 ! "
        "queue max-size-buffers=4 ! "
        "appsink name=sink emit-signals=false max-buffers=1 drop=true sync=false"
    )
    report_path = tmp_path / "gst-cpu-latest-smoke.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    result = _doctor_gst_cpu_latest_smoke_report(
        SimpleNamespace(report_json=str(report_path))
    )

    output = capsys.readouterr().out
    assert result == 2
    assert "missing: evidence.actual_pipeline_string has only latest-only queue elements" in output


def test_gst_cpu_latest_smoke_report_rejects_missing_metric_samples(tmp_path, capsys) -> None:
    report = _complete_report()
    report.pop("metric_samples")
    report_path = tmp_path / "gst-cpu-latest-smoke.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    result = _doctor_gst_cpu_latest_smoke_report(
        SimpleNamespace(report_json=str(report_path))
    )

    output = capsys.readouterr().out
    assert result == 2
    assert "missing: metric_samples nonempty list is required" in output


def test_gst_cpu_latest_smoke_report_rejects_incomplete_metric_sample(tmp_path, capsys) -> None:
    report = _complete_report()
    report["metric_samples"] = [{"elapsed_s": 1.0, "memory_rss_mb": 316.0}]
    report_path = tmp_path / "gst-cpu-latest-smoke.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    result = _doctor_gst_cpu_latest_smoke_report(
        SimpleNamespace(report_json=str(report_path))
    )

    output = capsys.readouterr().out
    assert result == 2
    assert "missing: metric_samples[0].capture_fps is required" in output
    assert "missing: metric_samples[0].actual_pipeline_string is required" in output
    assert "missing: metric_samples[0].inference_selected==tensorrt" in output


def test_gst_cpu_latest_smoke_report_rejects_sparse_metric_samples(tmp_path, capsys) -> None:
    report = _complete_report()
    samples = report["metric_samples"]
    assert isinstance(samples, list)
    first = dict(samples[0])
    last = dict(samples[0])
    first["elapsed_s"] = 0.0
    last["elapsed_s"] = 60.2
    report["metric_samples"] = [first, last]
    report_path = tmp_path / "gst-cpu-latest-smoke.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    result = _doctor_gst_cpu_latest_smoke_report(
        SimpleNamespace(report_json=str(report_path))
    )

    output = capsys.readouterr().out
    assert result == 2
    assert "missing: metric_samples cadence <=2.5s" in output


def test_gst_cpu_latest_smoke_report_cli_does_not_require_runtime_config(tmp_path, capsys) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        "capture:\n  backend: gst_cpu_latest\n  memory: nvmm\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "gst-cpu-latest-smoke.json"
    report_path.write_text(json.dumps(_complete_report()), encoding="utf-8")

    result = main([
        "--config",
        str(config_path),
        "doctor",
        "gst-cpu-latest-smoke-report",
        "--report-json",
        str(report_path),
    ])

    output = capsys.readouterr().out
    assert result == 0
    assert "accepted: True" in output


def test_gst_cpu_latest_smoke_command_writes_valid_report(tmp_path, monkeypatch, capsys) -> None:
    report_path = tmp_path / "gst-cpu-latest-smoke.json"
    pipeline_string = (
        "v4l2src device=/dev/video0 io-mode=2 do-timestamp=true ! "
        "queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream ! "
        "jpegparse ! nvv4l2decoder mjpeg=1 ! nvvidconv left=640 right=1280 top=220 bottom=860 ! "
        "video/x-raw,format=BGRx,width=640,height=640 ! "
        "queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream ! "
        "appsink name=sink emit-signals=false max-buffers=1 drop=true sync=false"
    )

    class FakePipeline:
        def __init__(self) -> None:
            self.started = False
            self.stopped = False

        def start(self) -> None:
            self.started = True

        def stop(self) -> None:
            self.stopped = True

        def status(self) -> dict[str, object]:
            return {
                "latest_frame_broker": {
                    "max_pending_depth": 1,
                    "published_frames": 7200,
                    "overwritten_frames": 100,
                    "acquired_frames": 7100,
                }
            }

    class FakeCapture:
        def __init__(self) -> None:
            self.configured = False
            self.stopped = False

        def configure(self, device: str, **_kwargs: object) -> SimpleNamespace:
            self.configured = True
            return SimpleNamespace(
                available=True,
                device=device,
                profile=SimpleNamespace(pixel_format="MJPG", width=1920, height=1080, fps=120),
                backend="gst_cpu_latest:nvmm-mjpg-iomode2",
                last_error="",
            )

        def stop(self, _reason: str) -> None:
            self.stopped = True

    class FakeRuntime:
        def __init__(self) -> None:
            self.pipeline = FakePipeline()
            self.last_inference_status = {
                "detection_batch_frame_id": 7164,
                "detection_batch_capture_ts_ns": 1_000_000,
                "inference_start_ts_ns": 1_004_000,
                "inference_end_ts_ns": 1_009_000,
                "frame_copy_cost_ms": 0.2,
                "detection_batch_model_input_size": [640, 640],
                "detection_coordinate_space": "roi",
            }
            self.last_control = {
                "frame_id": 7164,
                "control_now_ts_ns": 1_020_000,
            }

        def state(self) -> SimpleNamespace:
            return SimpleNamespace(
                capture={
                    "backend": "gst_cpu_latest:nvmm-mjpg-iomode2",
                    "statistics": {
                        "capture_fps": 119.4,
                        "published_frames": 7200,
                        "overwritten_frames": 100,
                        "acquired_frames": 7100,
                        "latest_frame_age_ms": 4.5,
                        "appsink_caps": "video/x-raw,format=BGRx,width=640,height=640",
                        "actual_pipeline_string": pipeline_string,
                    },
                },
                statistics={
                    "capture_fps": 119.4,
                    "published_frames": 7200,
                    "overwritten_frames": 100,
                    "acquired_frames": 7100,
                    "latest_frame_age_ms": 4.5,
                    "preprocess_ms": 1.3,
                    "h2d_ms": 0.4,
                    "inference_ms": 5.2,
                    "postprocess_ms": 0.1,
                    "batch_age_ms": 11.8,
                    "stale_drop_count": 3,
                    "control_observe_fps": 58.0,
                },
                inference={
                    "selected": "tensorrt",
                    "execution_backend": "tensorrt",
                    "device": "cuda",
                    "require_gpu": True,
                    "allow_cpu_fallback": False,
                    "loaded": True,
                    "available": True,
                    "preprocess_backend": "cpu",
                },
                pipeline=self.pipeline.status(),
            )

    fake_capture = FakeCapture()
    fake_runtime = FakeRuntime()
    fake_app = SimpleNamespace(
        state=SimpleNamespace(
            capture=fake_capture,
            runtime=fake_runtime,
            inference=SimpleNamespace(
                status=lambda: {
                    "selected": "tensorrt",
                    "loaded": True,
                    "available": True,
                    "device": "cuda",
                    "require_gpu": True,
                    "allow_cpu_fallback": False,
                }
            ),
        )
    )

    monkeypatch.setattr("novasight.main.create_app", lambda **_kwargs: fake_app)
    monotonic_values = iter([
        0.0,
        0.0,
        *[float(second) for second in range(1, 61)],
        60.2,
    ])
    monkeypatch.setattr(
        "novasight.main.time.monotonic",
        lambda: next(monotonic_values, 61.0),
    )
    monkeypatch.setattr("novasight.main.time.sleep", lambda _seconds: None)

    result = main([
        "--config",
        str(tmp_path / "missing.yaml"),
        "doctor",
        "gst-cpu-latest-smoke",
        "--device",
        "/dev/video0",
        "--seconds",
        "60",
        "--report-json",
        str(report_path),
    ])

    output = capsys.readouterr().out
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert result == 0
    assert "accepted: True" in output
    assert "metric_sample:" in output
    assert "capture_fps=119.40" in output
    assert "control_observe_fps=58.00" in output
    assert fake_capture.configured is True
    assert fake_capture.stopped is True
    assert fake_runtime.pipeline.started is True
    assert fake_runtime.pipeline.stopped is True
    assert report["accepted"] is True
    assert report["evidence"]["inference_selected"] == "tensorrt"
    assert report["evidence"]["latest_frame_broker_max_pending_depth"] == 1
    assert report["evidence"]["detection_batch"]["control_now_ts_ns"] == 1_020_000
    assert report["evidence"]["host_frame_copy_ms"] == 0.2
    samples = report["metric_samples"]
    assert isinstance(samples, list)
    assert samples
    assert samples[0]["elapsed_s"] >= 0.0
    assert samples[0]["capture_fps"] == 119.4
    assert samples[0]["published_frames"] == 7200
    assert samples[0]["host_frame_copy_ms"] == 0.2
    assert samples[0]["stale_drop_count"] == 3
    assert samples[0]["control_observe_fps"] == 58.0
    assert samples[0]["memory_rss_mb"] >= 0.0


def test_gst_cpu_latest_smoke_command_loads_explicit_tensorrt_engine(tmp_path, monkeypatch) -> None:
    report_path = tmp_path / "gst-cpu-latest-smoke.json"
    engine_path = tmp_path / "model.engine"
    engine_path.write_bytes(b"fake engine")
    loaded: dict[str, object] = {}
    pipeline_string = (
        "v4l2src device=/dev/video0 io-mode=2 do-timestamp=true ! "
        "queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream ! "
        "jpegparse ! nvv4l2decoder mjpeg=1 ! nvvidconv ! "
        "video/x-raw,format=BGRx,width=320,height=320 ! "
        "queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream ! "
        "appsink name=sink emit-signals=false max-buffers=1 drop=true sync=false"
    )

    class FakePipeline:
        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def status(self) -> dict[str, object]:
            return {
                "latest_frame_broker": {
                    "max_pending_depth": 1,
                    "published_frames": 10,
                    "overwritten_frames": 1,
                    "acquired_frames": 9,
                }
            }

    class FakeRuntime:
        def __init__(self) -> None:
            self.pipeline = FakePipeline()
            self.last_inference_status = {
                "detection_batch_frame_id": 10,
                "detection_batch_capture_ts_ns": 1_000_000,
                "inference_start_ts_ns": 1_004_000,
                "inference_end_ts_ns": 1_009_000,
                "frame_copy_cost_ms": 0.2,
                "detection_batch_model_input_size": [320, 320],
                "detection_coordinate_space": "roi",
            }
            self.last_control = {
                "frame_id": 10,
                "control_now_ts_ns": 1_020_000,
            }

        def state(self) -> SimpleNamespace:
            return SimpleNamespace(
                capture={
                    "backend": "gst_cpu_latest:nvmm-mjpg-iomode2",
                    "statistics": {
                        "capture_fps": 119.4,
                        "published_frames": 10,
                        "overwritten_frames": 1,
                        "acquired_frames": 9,
                        "latest_frame_age_ms": 4.5,
                        "appsink_caps": "video/x-raw,format=BGRx,width=320,height=320",
                        "actual_pipeline_string": pipeline_string,
                    },
                },
                statistics={
                    "capture_fps": 119.4,
                    "published_frames": 10,
                    "overwritten_frames": 1,
                    "acquired_frames": 9,
                    "latest_frame_age_ms": 4.5,
                    "preprocess_ms": 1.3,
                    "h2d_ms": 0.4,
                    "inference_ms": 5.2,
                    "postprocess_ms": 0.1,
                    "batch_age_ms": 11.8,
                    "stale_drop_count": 0,
                    "control_observe_fps": 58.0,
                },
                inference={
                    "selected": "tensorrt",
                    "execution_backend": "tensorrt",
                    "device": "cuda",
                    "require_gpu": True,
                    "allow_cpu_fallback": False,
                    "loaded": True,
                    "available": True,
                    "preprocess_backend": "cpu",
                },
                pipeline=self.pipeline.status(),
            )

    class FakeInference:
        def load(self, artifact_path, classes, input_shape) -> None:
            loaded["artifact_path"] = str(artifact_path)
            loaded["classes"] = list(classes)
            loaded["input_shape"] = input_shape

        def status(self) -> dict[str, object]:
            return {
                "selected": "tensorrt",
                "loaded": "artifact_path" in loaded,
                "available": True,
                "device": "cuda",
                "require_gpu": True,
                "allow_cpu_fallback": False,
            }

    fake_app = SimpleNamespace(
        state=SimpleNamespace(
            capture=SimpleNamespace(
                configure=lambda _device, **_kwargs: SimpleNamespace(
                    available=True,
                    backend="gst_cpu_latest:nvmm-mjpg-iomode2",
                    last_error="",
                ),
                stop=lambda _reason: None,
            ),
            runtime=FakeRuntime(),
            inference=FakeInference(),
        )
    )
    monkeypatch.setattr("novasight.main.create_app", lambda **_kwargs: fake_app)
    monotonic_values = iter([
        0.0,
        0.0,
        *[float(second) for second in range(1, 61)],
        60.2,
    ])
    monkeypatch.setattr(
        "novasight.main.time.monotonic",
        lambda: next(monotonic_values, 61.0),
    )
    monkeypatch.setattr("novasight.main.time.sleep", lambda _seconds: None)

    result = main([
        "--config",
        str(tmp_path / "missing.yaml"),
        "doctor",
        "gst-cpu-latest-smoke",
        "--device",
        "/dev/video0",
        "--seconds",
        "60",
        "--tensorrt-engine",
        str(engine_path),
        "--input-shape",
        "1x3x320x320",
        "--dtype",
        "fp16",
        "--classes",
        "enemy,head",
        "--report-json",
        str(report_path),
    ])

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert result == 0
    assert loaded == {
        "artifact_path": str(engine_path),
        "classes": ["enemy", "head"],
        "input_shape": "1x3x320x320",
    }
    assert report["parameters"]["tensorrt_engine"] == str(engine_path)
    assert report["evidence"]["tensorrt_engine"] == str(engine_path)


def test_gst_cpu_latest_smoke_command_rejects_explicit_engine_load_failure(
    tmp_path,
    monkeypatch,
) -> None:
    report_path = tmp_path / "gst-cpu-latest-smoke.json"
    engine_path = tmp_path / "model.engine"
    engine_path.write_bytes(b"fake engine")

    class FakeInference:
        def load(self, _artifact_path, classes, input_shape) -> None:
            del classes, input_shape
            pass

        def status(self) -> dict[str, object]:
            return {
                "selected": "tensorrt",
                "loaded": True,
                "available": True,
                "device": "cuda",
                "require_gpu": True,
                "allow_cpu_fallback": False,
                "last_switch_error": "deserialize failed",
            }

    fake_app = SimpleNamespace(
        state=SimpleNamespace(
            capture=SimpleNamespace(
                configure=lambda *_args, **_kwargs: SimpleNamespace(
                    available=True,
                    backend="gst_cpu_latest:nvmm-mjpg-iomode2",
                    last_error="",
                ),
                stop=lambda _reason: None,
            ),
            runtime=SimpleNamespace(
                pipeline=SimpleNamespace(
                    start=lambda: None,
                    stop=lambda: None,
                    status=lambda: {"latest_frame_broker": {"max_pending_depth": 1}},
                ),
                state=lambda: SimpleNamespace(capture={}, statistics={}, inference={}, pipeline={}),
                last_inference_status={},
            ),
            inference=FakeInference(),
        )
    )
    monkeypatch.setattr("novasight.main.create_app", lambda **_kwargs: fake_app)
    monotonic_values = iter([0.0, 61.0, 61.0])
    monkeypatch.setattr(
        "novasight.main.time.monotonic",
        lambda: next(monotonic_values, 61.0),
    )
    monkeypatch.setattr("novasight.main.time.sleep", lambda _seconds: None)

    result = main([
        "--config",
        str(tmp_path / "missing.yaml"),
        "doctor",
        "gst-cpu-latest-smoke",
        "--device",
        "/dev/video0",
        "--seconds",
        "60",
        "--tensorrt-engine",
        str(engine_path),
        "--report-json",
        str(report_path),
    ])

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert result == 2
    assert report["reason"] == "gst_cpu_latest_smoke_tensorrt_load_failed"
    assert report["detail"] == "deserialize failed"
