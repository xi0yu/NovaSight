# Capture Session Redesign

## Problem

The current implementation leaks internal engineering concepts into the product:

- The UI exposes "mainline" and "preview fallback" as user-facing ideas.
- `/api/capture/stream.mjpg` can open/read capture frames by itself when the runtime is not running.
- Capture, preview, and inference/control lifecycles are not clearly separated.
- Browser preview can accidentally become a second capture consumer or make users judge real capture performance from preview throughput.

This is the wrong product shape. A user should not need to understand fallback paths or start a hidden "mainline" before the camera behaves correctly.

## Goal

Create a single production-style capture session that owns the device handle and continuously produces latest frames. Preview and inference/control consume that session without owning or slowing capture.

## Core Principle

Browser streaming may be slow. Real capture and inference must not be slow because of it.

The capture session is the source of truth:

```text
/dev/video0
  -> CaptureSession thread
  -> latest frame cache / latest-frame queue
       -> MJPEG preview at 15/30/60 fps
       -> inference/control pipeline at capture pace
```

There must be no hidden preview-owned capture path in normal operation.

## Product Behavior

### Applying Capture Config

When the user applies a capture configuration:

1. Backend validates the requested device/profile.
2. Backend stops the previous capture session if it exists.
3. Backend opens the new configured GStreamer pipeline.
4. Backend starts the capture session immediately.
5. UI shows capture as `运行中`.

The user should not see or click "启动主线".

### Preview

Preview is a low-priority consumer:

- It reads only the latest frame produced by the capture session.
- It emits MJPEG at configured preview fps: `15`, `30`, or `60`.
- If preview cannot keep up, it drops preview frames only.
- Preview backpressure must never block capture.
- If no capture session is running, preview returns a clear non-200/empty-state response such as `采集会话未启动`.
- Preview must not open `/dev/video0`.

### Inference / Control

Inference/control is separate from capture:

- Capture can run without inference/control.
- Inference/control can be started/stopped independently.
- Starting inference/control consumes the current capture session.
- If capture is not running, starting inference/control fails fast with a clear message.

UI wording should be:

- Capture: `启动采集`, `停止采集`, `应用并启动采集`, `采集中`, `采集错误`
- Inference/control: `启动推理控制`, `停止推理控制`
- Preview: `预览输出`, `预览帧率`

UI must not use:

- `主线`
- `兜底`
- `fallback`
- `缓存消费者`

## Backend Design

### CaptureSession

Introduce or reshape a dedicated capture session abstraction. It should be the only component that reads from `FrameSource`.

Responsibilities:

- Own the active `FrameSource`.
- Own the capture thread.
- Continuously read frames as fast as the configured pipeline provides.
- Publish latest frame through a condition-protected cache.
- Maintain capture metrics.
- Fail fast or move into explicit error state on device/pipeline failure.
- Stop and close the device deterministically.

It should expose an interface like:

```python
class CaptureSession:
    def start(profile: CaptureProfile) -> CaptureRuntimeState: ...
    def stop(reason: str | None = None) -> CaptureRuntimeState: ...
    def reconfigure(profile: CaptureProfile) -> CaptureRuntimeState: ...
    def latest_frame(after_frame_id: int | None, timeout_s: float) -> CapturedFrame | None: ...
    def state() -> CaptureRuntimeState: ...
```

Exact names can follow existing code, but the ownership rule is strict: only the session reads from the source.

### CaptureService

`CaptureService` remains the API-facing coordinator:

- Query capabilities.
- Select profiles.
- Apply configuration.
- Start/stop/reconfigure `CaptureSession`.
- Return state.

It should not allow API routes to call source reads directly.

### MJPEG Route

`/api/capture/stream.mjpg` becomes a pure consumer:

- It never calls `capture.read_frame()`.
- It waits for `latest_frame()`.
- It encodes at preview fps.
- It records preview metrics separately.
- If capture is not running, it returns a clear status instead of starting capture.

### RuntimePipeline

`RuntimePipeline` should stop owning capture startup. It should assume capture already exists.

Start behavior:

- If capture session is running: attach to latest-frame flow.
- If capture session is not running: raise a clear error.

The pipeline can keep its latest-frame queue for inference/control, but frame supply comes from the capture session, not a second source read loop.

## Jetson Capture Path

The default production path for the tested capture card is:

```text
MJPG 1920x1080@120
  -> jpegparse
  -> nvv4l2decoder mjpeg=1
  -> nvvidconv
  -> video/x-raw(memory:NVMM),format=NV12,width=320,height=320
  -> appsink max-buffers=1 drop=true sync=false
```

Rules:

- Prefer explicit Jetson GStreamer pipelines.
- Do not silently fall back to OpenCV for the production Jetson path.
- Do not pull full-size BGR frames into Python for the real capture path.
- CPU BGR conversion is allowed only for preview rendering or controlled diagnostics, not as the default capture path.
- If required Jetson components are missing, fail with a clear diagnostic instead of hiding the issue behind a slow fallback.

## UI / UX Design

### Capture Workbench

The capture page should read like an operator console:

- Device selector.
- Capability groups by pixel format.
- Recommended profile.
- Action: `应用并启动采集`.
- State: `采集中`, `未启动`, `错误`.
- Metrics:
  - `采集帧率`
  - `采集等待`
  - `帧间隔`
  - `预览输出`
  - `预览丢帧`
  - `推理帧率` if available later

When capture is running:

- Config buttons are disabled or clearly require stopping capture first.
- The UI says `停止采集后可切换配置`.

### Preview Panel

Preview panel must not imply it is the real performance path.

Display:

- `预览输出 30fps`
- `浏览器预览低优先级，不影响采集与推理`

Avoid technical/internal wording.

### Runtime Controls

Separate controls:

- Capture control: `应用并启动采集`, `停止采集`
- Inference/control: `启动推理控制`, `停止推理控制`

No "mainline" wording.

## Error Handling

Use explicit failures over fallback chains.

Examples:

- Missing `gi`: `Jetson GStreamer Python bindings unavailable: install python3-gi for system Python or use system-site-packages venv.`
- Missing `nvv4l2decoder`: `Jetson MJPEG hardware decoder unavailable.`
- Device busy: `设备已被其他进程占用，无法启动采集。`
- No capture session for preview: `采集未启动，无法打开预览。`

Failures should be visible in UI and logs.

## Metrics

Separate metrics by responsibility:

- Capture metrics: produced by capture session.
- Preview metrics: produced by MJPEG route.
- Inference metrics: produced by inference/control pipeline.

Do not use preview fps as capture fps.

## Testing Strategy

Add focused tests, not many narrow tests:

- Applying a capture profile starts one capture session.
- MJPEG stream never calls `read_frame()` or opens source directly.
- Runtime inference/control start fails if capture is not running.
- Runtime inference/control consumes existing capture frames when capture is running.
- UI type/build checks cover renamed controls.
- Pipeline candidate tests preserve Jetson NVMM priority and no full-size CPU BGR default.

Manual Jetson verification:

1. Run backend.
2. Apply `MJPG 1920x1080@120`.
3. Confirm backend label is Jetson NVMM route.
4. Confirm capture fps is near device/GStreamer capability.
5. Open browser preview at 15/30fps.
6. Confirm capture fps does not drop because preview is open.
7. Start inference/control.
8. Confirm capture remains independent of preview.

## Non-Goals

- Full TensorRT integration in this change.
- KMNet final output protocol in this change.
- 100% test coverage.
- Multiple camera support.
- Generic cross-platform fallback capture stack.

## Implementation Decisions

Use these decisions in the implementation plan:

- Create `novasight/capture/session.py` for `CaptureSession`.
- Keep `CaptureService` as the API-facing coordinator.
- Let `RuntimePipeline` poll `latest_frame()` with frame-id tracking first; introduce subscriptions later only if profiling proves polling is a problem.
- Return `503` for preview when capture is not started, and let the UI show a Chinese empty state.
- Rename UI controls around product actions: capture controls manage capture, inference/control controls manage inference/control.
