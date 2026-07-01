# NovaSight Phase 2 Capture, Inference, and Control Design

Date: 2026-07-01

## Goal

Build the next NovaSight vertical slice around real Jetson capture, TensorRT
inference readiness, one migrated experimental vision/control plugin pair, and
a bounded control-output layer that can safely feed command-line inspection,
silent swallowing, and future kmNet hardware execution.

The priority for this phase is not a demo replay path. The main path is Linux
Jetson `/dev/video0` capture with automatic device capability discovery and a
chosen low-latency/high-fps configuration such as 1080p/144 MJPEG or 1080p/60
NV12 when the device supports it.

## Scope

In scope:

- Query `/dev/video0` or another V4L2 device with `v4l2-ctl`.
- Parse supported pixel formats, resolutions, and frame rates.
- Choose a capture profile automatically from device capabilities and runtime
  preference.
- Open a Jetson-oriented GStreamer capture pipeline and verify it by reading a
  real frame.
- Fall back to OpenCV V4L2 when GStreamer cannot open the selected profile.
- Expose capture state, selected profile, backend label, frame timing, and
  recent errors through runtime/API state.
- Add a TensorRT inference backend boundary that can load a released `.engine`
  artifact when TensorRT is available and report explicit unavailable state when
  it is not.
- Migrate one small experimental visual analysis plugin and one small control
  plugin as the template for later old-algorithm migration.
- Normalize and clamp `ControlIntent` values into configured output limits.
- Add `silent` and `console` control-output executors or executor modes before
  kmNet is wired to real hardware.
- Provide CLI diagnostics for camera capability and short capture smoke tests.

Out of scope:

- Full migration of every old `v3`, `straight`, `chris`, or `sunset` algorithm.
- Training, labeling, or model evaluation workflows.
- Real kmNet hardware send behavior.
- Production-quality video streaming optimization for browsers.
- Replacing the model registry with a new storage layer.

## Design Principles

Real capture is the first-class path. Replay/video fixtures remain useful for
tests, but the runtime should be designed around Jetson device discovery,
pipeline selection, and observable capture behavior.

Configuration stays runtime-only. YAML may describe source preference and output
limits, but it must not become the source of truth for model classes, plugin
internals, or algorithm-specific fields.

Hardware-facing behavior must be explicit. If TensorRT, GStreamer, `v4l2-ctl`,
or kmNet are unavailable, NovaSight should report that state plainly instead of
pretending a feature is active.

Control algorithms must remain isolated from hardware. Plugins return structured
control intent; executors and output policies own sending, printing, swallowing,
or later kmNet translation.

## Capture Architecture

Add a new capture package under `novasight/capture/`:

```text
novasight/capture/
  __init__.py
  caps.py
  profile.py
  pipeline.py
  source.py
  service.py
  state.py
```

`caps.py` owns V4L2 capability probing. It calls:

```bash
v4l2-ctl -d /dev/video0 --list-formats-ext
```

and parses output into structured records:

```python
CaptureCapability(
    pixel_format="MJPG",
    width=1920,
    height=1080,
    fps_list=[144, 120, 60, 30],
)
```

If `v4l2-ctl` is missing, the device is absent, or the command fails,
capability state is unavailable with a reason. This is not fatal for unit tests,
but the runtime should not claim a real camera profile was selected.

`profile.py` owns selection. It accepts capabilities and a preference:

- `auto_high_fps`: prefer highest frame rate, then width/height, then Jetson
  friendly formats.
- `auto_low_latency`: prefer NV12/YUYV raw formats when frame rate is close.
- `auto_balanced`: prefer 1080p or similar resolution before extreme frame rate.
- `manual`: require exact pixel format, width, height, and fps.

Default selection for this phase is `auto_high_fps`.

Format ranking for `auto_high_fps`:

1. `MJPG` when fps is at least 120, because many USB cameras expose high frame
   rates through compressed MJPEG and Jetson can decode via GStreamer/NVDEC.
2. `NV12` when fps is at least 60, because it is a strong low-copy path for
   capture cards and raw camera devices.
3. `YUYV` as a broad compatibility fallback.
4. Other formats only when no preferred format exists.

The selected profile is explicit:

```python
CaptureProfile(
    device="/dev/video0",
    pixel_format="MJPG",
    width=1920,
    height=1080,
    fps=144,
    selection_reason="auto_high_fps selected highest fps MJPG profile",
)
```

`pipeline.py` owns GStreamer string generation and backend labels. It should
generate candidates in priority order and test them by `open + read one frame`:

- `gst:nvmm-mjpg-iomode2`
- `gst:nvmm-mjpg-iomode4`
- `gst:nvmm-mjpg-ioauto`
- `gst:cpu-jpegdec-mjpg`
- `gst:nvmm-nv12-iomode2`
- `gst:nvmm-nv12-iomode4`
- `gst:nvmm-nv12-ioauto`
- `gst:cpu-nv12-videoconvert`
- `opencv:v4l2`

Only a candidate that successfully opens and reads a frame is selected. Failure
messages are kept in capture state for diagnostics.

`source.py` defines the frame source protocol and frame record. The source
should return width, height, pixel format, sequence number, monotonic timestamp,
capture wait duration, and image data. For Phase 2, BGR ndarray output is
acceptable even if the internal pipeline uses NV12 or MJPEG. Direct GPU tensor
handoff can be a later optimization.

`service.py` owns the capture loop. It updates:

- `fps_capture`
- `frame_period_ms`
- `capture_wait_ms`
- `frames_dropped`
- `recoveries`
- `last_error`
- selected device/profile/backend

It should tolerate transient `read()` failures, but repeated failures should
mark the source degraded and attempt bounded recovery rather than spinning.

## Runtime and API Integration

Runtime state should grow a `capture` section:

```json
{
  "capture": {
    "available": true,
    "device": "/dev/video0",
    "profile": {
      "pixel_format": "MJPG",
      "width": 1920,
      "height": 1080,
      "fps": 144
    },
    "backend": "gst:nvmm-mjpg-iomode2",
    "fps_capture": 141.8,
    "frame_period_ms": 6.95,
    "capture_wait_ms": 6.41,
    "frames_dropped": 0,
    "recoveries": 0,
    "last_error": null
  }
}
```

Add camera diagnostic endpoints:

- `GET /api/capture/capabilities?device=/dev/video0`
- `GET /api/capture/state`
- `POST /api/capture/select` for selecting a device/profile preference.

The Web UI can display this state, but Phase 2 does not require a large UI
redesign. A dense settings/status panel is enough.

## CLI Diagnostics

Add CLI subcommands while preserving `python3 -m novasight --host ... --port ...`
for the server:

```bash
python3 -m novasight doctor camera --device /dev/video0
python3 -m novasight capture-smoke --device /dev/video0 --seconds 5
```

`doctor camera` prints whether `v4l2-ctl` exists, whether the device can be
queried, the parsed profiles, and the profile NovaSight would choose.

`capture-smoke` opens the selected source, reads frames for the requested
duration, then reports observed FPS, average frame period, capture wait, backend
label, and recent failures.

These commands are intended for Jetson validation. Unit tests should mock the
shell runner and source reader so they still run on Mac.

## TensorRT Inference Boundary

Add `novasight/inference/`:

```text
novasight/inference/
  __init__.py
  contracts.py
  tensorrt.py
  unavailable.py
  runtime.py
```

The runtime dependency is an inference engine protocol:

```python
class InferenceEngine(Protocol):
    engine_id: str

    def available(self) -> bool: ...
    def load(self, artifact_path: Path, classes: list[str], input_shape: str) -> None: ...
    def infer(self, frame: CapturedFrame) -> InferenceResult: ...
```

`InferenceResult` should contain detections in the same coordinate space used by
`FrameContext`. This phase does not need a full tracker. If no TensorRT engine
is available, `UnavailableInferenceEngine` should return unavailable status and
the runtime should avoid claiming detections were produced.

The TensorRT implementation can be minimal but real:

- Import TensorRT and CUDA/PyCUDA or another selected runtime module lazily.
- Fail with a clear reason if imports are unavailable.
- Load only `.engine` artifacts published by the model registry.
- Validate that model classes and input shape exist on the active model version.
- Keep preprocessing/postprocessing behind small methods so model-specific
  behavior can be extended without leaking into runtime.

For local Mac tests, use a fake inference engine that returns deterministic
detections.

## Experimental Plugin Migration

Migrate one visual analysis plugin and one control plugin as templates, not as a
bulk copy of old algorithms.

Recommended pair:

- `vision.experimental_target`: selects the highest-quality target candidate
  and emits target metadata: center point, class name, score, normalized offset,
  and reason.
- `control.experimental_center`: consumes detections/tracks and emits
  `ControlIntent` toward the selected target.

Rules for migrated plugin code:

- No global YAML reads.
- No direct HID, kmNet, mouse, or keyboard imports.
- No hardware sends.
- No dependence on old package paths at runtime.
- Plugin configuration must be explicit and serializable.
- Coordinate behavior must be tested with fixed frame sizes and boxes.

The old project can be used as a reference for target selection and aiming
math, especially around center offset and class filtering, but Phase 2 should
not copy the old configuration architecture.

## Control Output Boundary

Add a control-output normalization layer before executors send anything outside
the process.

`ControlOutputPolicy` should clamp and scale raw `ControlIntent`:

```python
ControlOutput(
    dx=-120,
    dy=48,
    action="move",
    confidence=0.91,
    plugin_id="control.experimental_center",
    clipped=True,
    reason="clamped to max_abs_dx=120 max_abs_dy=120",
)
```

Runtime config owns only generic limits:

```yaml
control:
  max_abs_dx: 120
  max_abs_dy: 120
  min_confidence: 0.25
  output_mode: silent
```

Supported output modes for Phase 2:

- `silent`: accepts and records bounded output, but performs no external action.
- `console`: writes one concise line per bounded output for command-line
  inspection.
- `dry_run`: preserve existing executor semantics, but route through the same
  bounded output policy.

This prepares the future `kmnet` executor: kmNet should consume
`ControlOutput`, not raw plugin intent.

## Error Handling

Capture errors are classified:

- capability unavailable: `v4l2-ctl` missing, command failed, or no formats.
- profile unavailable: manual profile not supported by device.
- backend unavailable: GStreamer/OpenCV cannot open or read selected profile.
- runtime degraded: source opened but repeated reads fail.

Inference errors are classified:

- no active model deployment.
- active artifact is not an engine.
- TensorRT runtime unavailable.
- engine load failed.
- inference failed for a frame.

Control output errors are classified:

- confidence below threshold.
- non-finite dx/dy.
- output clipped.
- selected output mode unavailable.

Errors should be observable in state and API responses. Frame-level transient
errors should not crash the server; programmer errors should still fail tests.

## Testing Strategy

Tests must run on Mac without camera, TensorRT, or kmNet:

- Parse representative `v4l2-ctl --list-formats-ext` output.
- Select `MJPG 1920x1080@144` over lower-fps alternatives for
  `auto_high_fps`.
- Select `NV12 1920x1080@60` over `MJPG 1920x1080@30` when low-latency
  preference is requested.
- Reject unsupported manual profiles with clear errors.
- Generate expected GStreamer candidate labels and caps for MJPG and NV12.
- Choose the first backend candidate that opens and reads a frame.
- Fall back to OpenCV V4L2 when GStreamer candidates fail.
- Update capture timing state from a fake frame source.
- Report capability/capture state through API.
- Report TensorRT unavailable state without importing TensorRT at module import
  time.
- Convert fake inference detections into plugin `FrameContext`.
- Verify experimental plugin coordinate math.
- Clamp, drop, or pass control output according to configured limits.
- Verify console/silent output modes do not call hardware.

Jetson manual validation:

```bash
python3 -m novasight doctor camera --device /dev/video0
python3 -m novasight capture-smoke --device /dev/video0 --seconds 5
python3 -m novasight --host 127.0.0.1 --port 5174
```

For a successful Jetson capture smoke, the report should include the selected
profile, backend label, observed FPS, frame period, capture wait, and zero or
bounded recovery attempts.

## Implementation Order

1. Capture capability parser and profile selector.
2. GStreamer/OpenCV source candidate generation and source opening contract.
3. Capture service state and CLI diagnostics.
4. API integration for capture capabilities/state/select.
5. TensorRT inference boundary and unavailable/fake engines.
6. Runtime frame flow from captured frame to inference result to plugins.
7. Experimental visual/control plugin pair.
8. Control output policy and `silent`/`console` modes.
9. Web console capture status panel.
10. Documentation and Jetson smoke instructions.

This order keeps hardware discovery first, then inference, then algorithm
migration, then output safety. Each part is independently testable.

## Spec Self-Review

- No placeholders remain.
- Capture is explicitly Jetson-first and `/dev/video0`-oriented.
- TensorRT unavailable behavior is explicit and testable.
- Control plugins remain isolated from hardware.
- Control output is bounded before any future kmNet execution.
- The scope is large but coherent for one Phase 2 plan because each subsystem
  forms one end-to-end vertical slice.
