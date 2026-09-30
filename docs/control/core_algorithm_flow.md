# NovaSight Core Algorithm Flow

Date: 2026-09-30

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
-> hardware trigger hold threshold
-> aim-point velocity prediction
-> continuous Atan control
-> per-axis output limit
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

In hardware-trigger mode the control worker does not invoke the prediction or
control algorithm until the observed button hold duration exceeds the configured
fire delay. Short presses therefore never create an `OutputPlan`.

The consequence is simple: every control command should represent the freshest
available target state, not a replay of old frames.

## 3. Target Tracking And Selection

The targeting worker consumes the newest `DetectionBatch`.

It uses the observation geometry center as the control reference point.

Then it selects and tracks targets:

```text
detections
-> class/confidence/aspect-ratio admission
-> observed aim inside the search circle (selection eligibility)
-> bounded track association
-> stable TrackId
-> target ranking
-> selected aim_x / aim_y
```

The tracker uses a bounded constant-velocity Kalman state for identity
association. That Kalman state is for keeping the same target identity across
frames. It is not the final mouse-control prediction.

The only geometric selection gate is `distance(observed_aim, center) <= R`,
where `R = target_fov_radius_px` in observation pixels. The aim point uses the
configured class X/Y ratios, not necessarily the box center. Box overlap with
the circle is insufficient; predicted positions cannot admit an outside aim.
Box size does not scale the radius. Outside observations may retain an identity,
but never authorize control output. This is post-inference candidate filtering,
not a capture or inference crop.

Acquisition first checks the current observed aim point's distance to the
observation center. A point is considered nearby when its distance is no more
than half the detection box's shorter side. This is an initial local-selection
heuristic requiring recorded-scene and hardware evaluation, not an estimate of
user intent. It is independent of the search radius and never admits an aim
outside that circle. Among nearby points, smaller distance wins; the score below
breaks exact distance ties. Box containment does not define a preferred part.

When no nearby point exists, acquisition uses three current-observation signals:

```text
L = hypot(observation_width / 2, observation_height / 2)
D = 1 - clamp(distance_to_observed_aim / L, 0, 1)
C = configured class weight (unlisted classes: 0)
Q = current detector confidence
score = (w_distance*D + w_class*C + w_confidence*Q) / sum(weights)
```

Box size defines only the local-selection neighborhood; it does not add score
or enlarge the search circle. Association continuity and motion direction do
not add merit. A confirmed, currently observed lock inside the search circle
keeps its choice regardless of another candidate's class score. Hardware trigger
release/repress clears the choice without deleting tracked identities. Always-on
mode keeps a valid choice until it becomes unavailable or the pipeline resets.
An unavailable lock emits no target; a fallback must pass continuity and
capture-time confirmation. The old preference-advantage field is retained for
configuration compatibility, but no longer controls switching.

Class preference remains soft for distant acquisition: a higher class weight
does not guarantee selection over distance and confidence. Changing the search
radius changes eligibility only, not rank among already admitted candidates.

Cross-class association requires box IoU of at least 0.5 before applying the
existing penalty. This conservatively separates nested part boxes while still
allowing a label flicker on substantially the same box; it does not prove that
two parts belong to the same person. Its identity-quality
score uses geometry only, so increasing the class penalty cannot manufacture
confidence for a same-class match. This quality score is a heuristic, not a
calibrated probability. A cold/lagging Kalman prediction may fall back to the
last observed center only if it passes the same NIS outlier bound. Neither
reference passing means no association; the hard threshold is not relaxed.

Kalman prediction retention is bounded by elapsed capture time, without a
second frame-count limit or hidden multiplicative confidence-decay expiry.
Finite-state, covariance, position-uncertainty, NIS, and identity-quality guards
remain and can reject prediction before that time limit. A retained/lost track
never becomes a control target without a current admitted detection.

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

The offset is accepted only for valid timing, finite arithmetic and consistent motion. Valid displacement is not clipped to a fixed pixel radius. Stops, reversals and unavailable history still disable extrapolation. Final X/Y device-count limits remain independent of prediction.

The device delay is a measured calibration value, not a second strength knob. Extra lead adds time; zero extra lead still compensates observation age and device delay.

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
S = 256 counts
q = clamp((t - entry_start) / entry_ramp_ms, 0, 1)
a = 3*q*q - 2*q*q*q; entry_ramp_ms=0: a=1
demand = a * response_scale * S * atan(error_counts / S)
limit_x = max_output_x_counts
limit_y = max_output_y_counts
```

This means:

```text
small error -> approximately proportional response
large error -> compressed by Atan, no additional gain boost
entry -> smoothly reach configured Kp within entry_ramp_ms
```

Entry restarts after target/gate changes or capture gaps exceeding an internal
80ms continuity budget (the previous default). Frame age is a separate freshness
check: a new frame arriving every 60ms is not necessarily 60ms old. Changing
prediction-history retention no longer changes the entry envelope, including
when prediction is disabled. Low-rate control below 12.5Hz requires revisiting
this existing continuity policy; it is not claimed as validated here.

There is no traditional PID loop in the current mainline:

```text
no Ki accumulation
no explicit D term
no prediction of mouse counts
```

## 6. Output Limit And Integer Conversion

The controller quantizes demand first; the device worker owns the final
output-bound policy:

```text
float demand
-> carry fractional remainder needed by integer-only hardware
-> integer dx / dy
-> device worker rechecks current gates and generation
-> clamp integer dx / dy to the configured per-axis output bounds
```

The fractional remainder is internal conversion state, not a parameter and not
a stop condition. There is no arrival radius, arrival hysteresis, visual
feedback wait, or suppression of repeated `+1` / `-1` corrections.

## 7. Device Output

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

The device worker applies the final X/Y limits to the tracking command.
A zero movement never produces a physical send.

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

## Schema 19 Search Range Migration

`target_range_scale` is retired from runtime, configuration schema, and Studio.
The existing `target_fov_radius_px` value is preserved (180 px if missing).
There is no percentage-to-pixel conversion: capsule scale depended on each
target's dimensions and was not equivalent to a fixed search radius. Loading
migrates in memory; the next ordinary save writes schema 19 and removes the
retired key. Explicit writes to that key are rejected.

Removing the capsule gate intentionally admits more targets within the existing
search circle. The distance score now uses the observation half-diagonal rather
than the search radius, so older weights can choose differently. This is not a
behavior-equivalent migration. Keep a configuration backup when deploying and
verify admission and ranking before enabling physical output. No running device
configuration is changed by editing this repository.

## Schema 18 Cleanup And Verification Boundary

Removed from runtime config, API schema, and Studio controls:

```text
target_selection_size_weight
target_selection_continuity_weight
target_selection_motion_weight
target_selection_motion_horizon_ms
tracker_kalman_max_predict_steps
target_class_priority
```

Old class order is converted once into explicit class weights; existing explicit
weights override the migrated rank values. Empty modern weight maps have no
implicit class preference. Removed scoring fields are discarded during migration.
If an old profile exclusively used removed merits, it receives the default three
remaining weights. Malformed active settings and unknown keys still fail
validation. Migration is idempotent and persists with the normal configuration
save; loading alone does not overwrite the user's file.

Hardware calibration, trigger delay, capture-age validation, and
final device limits remain. Valid predictions have no fixed pixel cap. Final output still has independent X/Y bounds; this
pass does not claim a vector-only device limiter or change hardware semantics.
Host simulations establish deterministic contracts, not Jetson accuracy or a
universally optimal parameter set. Real capture latency, detector noise, and
physical-device response still need measurement.

Schema 20 uses one internal capture-age budget, `pipeline.frame_max_age_ms`
(default/maximum 50 ms), for GPU admission, post-inference admission, control and
device-worker dispatch. The age is always measured from the original capture,
not from a previous gate. Latest generation alone does not prove freshness.
Legacy deadlines migrate to the strictest of the old values and 50 ms. Detection
and target selection share `inference.confidence_threshold`; migration keeps the
higher old threshold. Reading migrates in memory; normal saving persists it.
The three scoring weights and switch/lost-identity timers remain internal and
are not consumer controls. Per-class preference and aim points remain editable.

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
