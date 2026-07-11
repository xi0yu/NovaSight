from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

router = APIRouter(prefix="/api/device", tags=["device"])


@router.get("/capabilities")
def device_capabilities(request: Request) -> dict[str, Any]:
    """Describe this NovaSight node for remote Studio clients.

    The response is deliberately about product/runtime capabilities, not raw
    hardware inventory. Studio uses this to decide whether the connected node is
    a Jetson execution endpoint and which remote-management features to enable.
    """
    config = request.app.state.runtime.config_store.snapshot()
    runtime_state = request.app.state.runtime.state()
    inference_status = dict(runtime_state.inference)
    capture = request.app.state.capture
    capture_state = capture.state
    capture_session = getattr(capture, "session", None)
    executor_status = request.app.state.executors.status()
    capture_backend = str(getattr(config.capture, "backend", "")).lower()
    inference_backend = str(getattr(config.inference, "backend", "")).lower()
    capture_memory = str(config.capture.memory).lower()
    return {
        "node_role": "jetson_runtime",
        "studio_role": "remote_manager",
        "realtime_owner": "jetson",
        "critical_path": {
            "host_in_realtime_loop": False,
            "jetson_runs_capture": True,
            "jetson_runs_inference": True,
            "jetson_runs_control": True,
        },
        "transports": {
            "rest": True,
            "websocket": True,
            "grpc": False,
        },
        "endpoints": {
            "health": "/healthz",
            "status_ws": "/ws/status",
            "runtime_state": "/api/runtime/state",
            "runtime_start": "/api/runtime/start",
            "runtime_stop": "/api/runtime/stop",
            "config": "/api/config",
            "config_schema": "/api/config/schema",
            "capture_select": "/api/capture/select",
            "capture_preview": "/api/capture/stream.mjpg",
            "model_scan": "/api/models/scan",
        },
        "features": {
            "model_management": True,
            "parameter_configuration": True,
            "logs_and_performance": True,
            "remote_runtime_control": True,
            "onnx_training_or_export": False,
            "full_deepstream_pipeline_generation": inference_backend == "deepstream_nvinfer",
            "gst_cpu_latest_capture": capture_backend == "gst_cpu_latest",
            "deepstream_capture_data_plane": capture_memory == "nvmm",
            "deepstream_object_meta_control": inference_backend == "deepstream_nvinfer",
            "custom_tensorrt_scheduler": inference_backend in {"tensorrt", "nvmm_latest"},
            "nvmm_capture_configured": capture_memory == "nvmm",
            "gpu_resource_preprocess_available": bool(
                inference_status.get("gpu_preprocessor", {}).get("available", False)
            ),
            "kmnet_output_available": str(executor_status.get("selected", "")).lower() == "kmnet",
        },
        "active_config": {
            "capture_device": config.capture.device,
            "capture_backend": config.capture.backend,
            "capture_memory": config.capture.memory,
            "preprocess_backend": config.preprocess.backend,
            "freshness_threshold_ms": config.runtime.freshness_threshold_ms,
            "roi_size": config.roi.size,
            "preview_fps": config.limits.stream_fps,
            "output_mode": str(executor_status.get("selected", "kmnet") or "kmnet"),
            "inference_backend": config.inference.backend,
        },
        "runtime": {
            "capture_available": bool(getattr(capture_state, "available", False)),
            "capture_running": (
                bool(runtime_state.running)
                if inference_backend == "deepstream_nvinfer"
                else bool(getattr(capture_session, "running", False))
            ),
            "inference_loaded": bool(inference_status.get("loaded", False)),
            "inference_engine": str(inference_status.get("selected", "")),
        },
        "mainline": {
            "backend": (
                "gst_cpu_latest_custom_tensorrt"
                if capture_backend == "gst_cpu_latest"
                else "deepstream_nvmm_nvinfer_object_meta"
                if inference_backend == "deepstream_nvinfer"
                else "deepstream_capture_custom_tensorrt"
            ),
            "capture_memory": config.capture.memory,
            "capture_backend": config.capture.backend,
            "preprocess_backend": config.preprocess.backend,
            "inference_backend": config.inference.backend,
        },
    }
