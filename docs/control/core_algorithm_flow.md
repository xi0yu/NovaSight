# NovaSight Core Algorithm Flow

Date: 2026-08-04

This document describes the current production algorithm path. It is written as
the runtime flow a user would experience on a Jetson, not as crate-by-crate API
documentation.

## One-Line Flow

```text
DeepStream capture/inference
-> DetectionBatch
-> latest-only ingress
-> target tracking and selection
-> selected aim_x / aim_y
-> aim-point velocity prediction
-> continuous Atan control
-> per-axis output limit
-> recoil mix
-> kmNet / HID output
```

## 1. Capture And Inference

The live product path starts from the DeepStream runtime. DeepStream owns camera
capture, GPU preprocessing, TensorRT inference, and object metadata extraction.
Rust does not keep borrowed NVIDIA/GStreamer pointers after metadata extraction.
The native seam hands Rust an owned detection snapshot.

The important output of this stage is a `DetectionBatch`:

```text
frame epoch
generation
capture timestamp
ROI / observation dimensions
detections: bbox, class_id, confidence, object_id
```

This batch is already the unit of truth for downstream targeting and control.

## 2. Latest-Only Runtime Lanes

After inference, NovaSight does not build a long frame queue. The runtime uses
latest-only slots:

```text
DetectionBatch slot
-> TargetedObservation slot
-> OutputPlan slot
```

If a newer frame arrives while a worker is behind, the unread older item is
replaced. This is intentional: stale control is worse than dropped control.

The three active post-inference workers are:

```text
targeting worker
control worker
device worker
```

The consequence is simple: every control command should represent the freshest
available target state, not a replay of old frames.

## 3. Target Tracking And Selection

The targeting worker consumes the newest `DetectionBatch`.

It first resolves the control reference point:

```text
configured / learned crosshair
or geometric center fallback
```

Then it selects and tracks targets:

```text
detections
-> class/confidence/FOV admission
-> bounded track association
-> stable TrackId
-> target ranking
-> selected aim_x / aim_y
```

The tracker uses a bounded constant-velocity Kalman state for identity
association. That Kalman state is for keeping the same target identity across
frames. It is not the final mouse-control prediction.

The control aim point comes from the current selected detection box and class
aim ratio. For example, different classes can aim at a different vertical point
inside the bbox.

## 4. Aim-Point Prediction

Prediction is based on target aim-point motion, not mouse error and not output
counts.

For each selected target:

```text
P_i = (aim_x_i, aim_y_i)
t_i = capture timestamp
```

NovaSight keeps a small four-position history for the current target. From that
it derives adjacent velocities:

```text
vx = dx / dt
vy = dy / dt
```

The runtime uses real capture timestamps for `dt`. It does not assume a fixed
FPS.

The current prediction horizon is:

```text
horizon_ms = frame_age_ms + actuation_delay_ms + prediction_lead_ms
```

Then:

```text
prediction_velocity = vector_medoid(v1, v2, v3)
predicted_offset = prediction_velocity * horizon_ms
```

The offset is then gated and capped:

```text
vector-medoid velocity-time offset
-> prediction_cap_px vector cap
-> safe prediction offset
```

This cap belongs to prediction only: it limits how far the target aim point can
be advanced. It is separate from the later device-count output limit.

Acceleration is telemetry only in the current model. It does not add a second
prediction correction path.

## 5. Continuous Atan Control

The controller receives:

```text
selected aim_x / aim_y
crosshair_x / crosshair_y
safe prediction offset
trigger state
frame age
track confidence
```

It computes measured error:

```text
error_px = aim - crosshair
```

Then prediction changes the control point:

```text
control_error_px = measured_error_px + safe_prediction_offset_px
```

That error is projected into device counts through the configured geometry:

```text
ROI/source dimensions
FOV
counts_per_360
```

The proportional response is a continuous Atan response:

```text
rho = hypot(error_counts_x, error_counts_y)
S = 256 counts
r = rho / S
curve = 1 - exp(-(r ^ response_curve_shape))
R = 1 + response_boost * curve
gain = response_scale * R
demand = gain * S * atan(error_counts / S)
limit_x = max_output_x_counts
limit_y = max_output_y_counts
```

This means:

```text
small error -> softer response
middle error -> continuous transition
large error -> stronger response, still compressed by Atan
```

There is no traditional PID loop in the current mainline:

```text
no Ki accumulation
no explicit D term
no prediction of mouse counts
```

## 6. Output Limit And Integer Conversion

The limiter owns the only output-bound policy and the final integer mouse
counts:

```text
out_x = clamp(float_demand_x, -max_output_x_counts, max_output_x_counts)
out_y = clamp(float_demand_y, -max_output_y_counts, max_output_y_counts)
-> carry fractional remainder needed by integer-only hardware
-> integer dx / dy
```

The fractional remainder is internal conversion state, not a parameter and not
a stop condition. There is no arrival radius, arrival hysteresis, visual
feedback wait, or suppression of repeated `+1` / `-1` corrections.

## 7. Recoil And Device Output

The device worker receives the latest `OutputPlan`.

Before sending, it rechecks:

```text
runtime still running
output gate open
device connected
hardware trigger still active if required
command epoch
command generation is still latest
```

Then recoil can be mixed into the Y axis:

```text
tracking command
+ interval recoil command
-> one physical send
```

If a newer frame has already superseded the command, the device worker drops the
old command. This preserves latest-only behavior all the way to hardware output.

The final output is sent through the configured pointer device path, currently
kmNet/HID on production Jetson builds.

## What The Current Algorithm Is Not

```text
not fixed PID
not Ki/D tuning
not a multi-frame command queue
not frame-count-based prediction lead
not app-layer Kalman prediction for mouse output
not prediction directly in counts
```

The core model is:

```text
target aim motion
+ time lead
+ continuous nonlinear P response
+ X/Y output limit
= one latest physical output command
```

## Performance Shape

The current design keeps Jetson-side application work bounded:

```text
single selected target for control prediction
four aim-position samples for velocity
bounded association surface
latest-only slots instead of queues
no heavy app-layer filter stack
no repeated old-frame control playback
```

The expensive work should remain in DeepStream/TensorRT. Rust should keep owning
validation, target state, prediction math, control math, output safety, and UI
telemetry.

## Remaining Rust Test Surface

The Rust tests intentionally kept after cleanup cover:

```text
core geometry / freshness / tracking / output contracts
control trace parity and runtime algorithm slices
prediction and response unit tests
DeepStream owned snapshot admission
DeepStream pipeline string contracts
one pipeline runtime detection-to-output slice
runtime config composition
config repository migration
```

Deleted test groups were mostly outer product shell coverage:

```text
API route matrix
CLI help/command snapshots
UDS client route parity
runtime supervisor lifecycle matrix
license/model catalog repository regression matrix
extra pipeline safety duplicates
```

Those areas can still be verified manually or with focused tests when they are
being changed, but they should not dominate everyday algorithm iteration.
