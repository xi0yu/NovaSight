from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from novasight.deepstream import check_deepstream_dependencies

router = APIRouter(prefix="/api/device", tags=["device"])


@router.get("/capabilities")
def device_capabilities(request: Request) -> dict[str, Any]:
    """Describe this NovaSight node for remote Studio clients.

    The response is deliberately about product/runtime capabilities, not raw
    hardware inventory. Studio uses this to decide whether the connected node is
    a Jetson execution endpoint and which remote-management features to enable.
    """
    config = request.app.state.runtime.config_store.snapshot()
    inference_status = request.app.state.inference.status()
    capture = request.app.state.capture
    capture_state = capture.state
    capture_session = getattr(capture, "session", None)
    executor_status = request.app.state.executors.status()
    deepstream_status = check_deepstream_dependencies()
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
            "deepstream_pipeline_generation": True,
            "deepstream_runtime_backend": deepstream_status.available,
            "deepstream_realtime_path_selected": str(config.inference.backend).lower() == "deepstream",
            "legacy_nvmm_capture_configured": str(config.capture.memory).lower() == "nvmm",
            "nvmm_capture_configured": str(config.capture.memory).lower() == "nvmm",
            "gpu_resource_preprocess_available": bool(
                inference_status.get("gpu_preprocessor", {}).get("available", False)
            ),
            "kmnet_output_available": str(executor_status.get("selected", "")).lower() == "kmnet",
        },
        "active_config": {
            "capture_device": config.capture.device,
            "capture_memory": config.capture.memory,
            "roi_size": config.roi.size,
            "preview_fps": config.limits.stream_fps,
            "output_mode": config.control.output_mode,
            "inference_backend": config.inference.backend,
            "deepstream_manifest_path": config.inference.deepstream_manifest_path,
            "deepstream_config_path": config.inference.deepstream_config_path,
            "deepstream_io_mode": config.inference.deepstream_io_mode,
            "deepstream_batched_push_timeout_us": config.inference.deepstream_batched_push_timeout_us,
        },
        "runtime": {
            "capture_available": bool(getattr(capture_state, "available", False)),
            "capture_running": bool(getattr(capture_session, "running", False)),
            "inference_loaded": bool(inference_status.get("loaded", False)),
            "inference_engine": str(inference_status.get("selected", "")),
        },
        "deepstream": {
            "available": deepstream_status.available,
            "reason": deepstream_status.reason,
            "detail": deepstream_status.detail,
        },
    }
