# NovaSight AI Purpose Audit Questions

This file is for AI agents auditing whether the current project actually reaches
NovaSight's intended runtime purpose. It is not a human-facing design overview.

Use these questions to force evidence. Do not answer from file names, TODOs, or
claims in docs alone. Prefer code paths, config validation, tests, runtime status
payloads, and Jetson measurement logs.

## How To Answer

For each question, answer with one of:

- `YES`: production code reaches this behavior and there is evidence.
- `PARTIAL`: the module exists, but it is opt-in, isolated, unverified, or not wired into the production path.
- `NO`: the behavior is absent or contradicted by production code.
- `UNKNOWN`: the answer requires unavailable hardware, logs, or runtime data.

Every answer should include:

- Evidence files and functions.
- The active production entry path.
- Tests or commands that prove the answer.
- Missing evidence or the next smallest check.

## Core Runtime Purpose

1. Is the current project actually using DeepStream as the active detection path, rather than only `onnxruntime` or Python TensorRT?

   Evidence to inspect:
   - `RuntimeConfig.inference.backend` allowed values.
   - The startup path that creates the active capture/inference/runtime objects.
   - Whether `DeepStreamDetectionBackend` is instantiated by the production runtime.
   - Whether `/api` or CLI can select DeepStream for live runtime execution.
   - Runtime status fields showing the selected backend.

   A `YES` answer requires the live production path to be:

   ```text
   v4l2src
   -> NVIDIA decode / NVMM
   -> nvvidconv ROI/resize
   -> nvstreammux
   -> nvinfer
   -> NvDsInferTensorMeta
   -> DetectionBatch
   -> Tracker/Control
   ```

   If DeepStream code only builds a pipeline string, generates `deepstream.ini`,
   or exposes an experimental backend that is not selected by runtime startup,
   answer `PARTIAL`, not `YES`.

2. Does the production path avoid using `gst-launch-1.0` as the runtime backend?

   A `YES` answer means `gst-launch-1.0` is only used for diagnostics or manual
   verification. Production runtime should create and own the GStreamer pipeline
   in process, with lifecycle, status, and error handling.

3. Does the project distinguish these three layers clearly?

   ```text
   GStreamer pipeline validation
   DeepStream detection source
   ONNXRuntime/TensorRT inference engine
   ```

   A `NO` or `PARTIAL` answer is likely if config treats DeepStream as just
   another model engine while the actual implementation owns capture and decode.

4. Is `DetectionBatch` the single interface consumed by downstream tracking and control?

   Inspect whether downstream modules depend only on NovaSight contracts, not
   DeepStream, GStreamer, `pyds`, TensorRT buffers, or raw model tensors.

5. Can the runtime switch between legacy detection and DeepStream detection
   without changing tracker, target selector, Kalman, aim, controller, scheduler,
   or device adapter code?

   A `YES` answer requires a real source/interface seam. A conditional spread
   across downstream modules is `NO`.

## DeepStream And NVMM Evidence

6. Does the active DeepStream path keep frames in NVMM or GPU-side memory until
   tensor output, rather than mapping full frames into CPU BGR/RGB arrays?

   Evidence should include pipeline caps, runtime status, or Jetson logs. A
   generated pipeline string alone is not enough unless it is the active runtime
   pipeline.

7. Does the pipeline use NVIDIA hardware decode for the selected camera format?

   Check format routing for `MJPG`, `YUYV`, `NV12`, `H264`, and `H265`. If only
   MJPEG is implemented, answer `PARTIAL`.

8. Does the project query real V4L2 device capabilities before selecting format,
   resolution, and FPS?

   Evidence to inspect:
   - `v4l2-ctl --list-formats-ext` wrapper or equivalent.
   - Parsed capabilities.
   - UI/API/config validation that prevents unsupported selections.

9. Is ROI crop calculated from capture resolution, ROI size, and offsets before
   model resize?

   The answer should mention the coordinate spaces involved:

   ```text
   capture -> ROI -> model -> ROI/control
   ```

10. Is DeepStream `nvinfer` configured to output tensor meta, not default bbox
    parser output?

    A `YES` answer should show `output-tensor-meta=1` and code that reads
    `NvDsInferTensorMeta`.

11. Does tensor meta extraction preserve the model output shape and dtype expected
    by the manifest?

    If this has not been validated on Jetson with `pyds`, answer `UNKNOWN` or
    `PARTIAL`.

12. Does the DeepStream path reuse the shared YOLO parser instead of duplicating
    bbox decode/NMS logic?

13. Does the DeepStream path publish measured `DetectionBatch` FPS, tensor meta
    FPS, frame age, and postprocess latency?

    Do not accept `fpsdisplaysink` as sufficient evidence.

## Model Configuration Purpose

14. Does every TensorRT engine used by DeepStream have a persistent model
    manifest?

   A `YES` answer requires durable metadata for input shape, output shape, class
   count, parser semantics, thresholds, artifact hash, and config fingerprint.

15. Does the project avoid guessing irreversible model semantics at every startup?

   If shape or class count must be manually confirmed once and then cached, this
   is good. If runtime guesses silently every boot, answer `NO`.

16. Is `deepstream.ini` generated deterministically from the manifest and model
    fingerprint?

17. Does the model registry detect these states?

   ```text
   ready
   need_confirm
   invalid
   unsupported
   ```

18. Can a model switch be done safely without mixing old tensor meta, old tracker
    state, and new model outputs?

   A `YES` answer requires stop/pause, flush, clear state, load, restart, wait
   first tensor meta, and then mark active.

## Algorithm And Control Purpose

19. Does the control chain use this geometry, rather than direct pixel-domain PD?

   ```text
   error_px
   -> error_rad
   -> angular PD
   -> calibrated counts
   -> scheduler
   -> device adapter
   ```

20. Is `pixel_linear`, legacy dynamic PID, or direct `Kp * error_px` prevented
    from production execution?

   Search both config validation and runtime execution paths. Deleted files are
   not enough if old behavior remains under a new name.

21. Does control use a calibration profile with `fov_x_deg`,
    `counts_per_360_x/y`, axis signs, and sensitivity fingerprint?

22. Does a calibration or sensitivity mismatch reset runtime control state and
    block unsafe output?

23. Does the tracker separate candidate filtering, quality scoring, association,
    target selection, and control output?

24. Does the tracker handle identity uncertainty as a control state, rather than
    instantly switching target?

25. Does Kalman prediction have hard limits for missing time, covariance, NIS,
    prediction steps, and identity confidence?

26. Does latency compensation avoid double-predicting old measurements?

27. Does the command scheduler cancel or expire stale commands on new frames,
    direction changes, target changes, and device errors?

28. Does the device adapter receive only scheduled commands, not raw detections,
    raw bboxes, or raw pixel errors?

## Time, Logging, Replay, And Evidence

29. Does the runtime use monotonic timestamps for capture, inference, tracking,
    latency compensation, command TTL, and frame age?

30. Are wall-clock timestamps kept out of control math?

31. Can a recorded run be replayed through detection/tracking/control decisions
    without the live camera or live device?

32. Does the runtime log enough data to diagnose why a command was or was not
    emitted?

   Minimum useful fields:

   ```text
   frame_id
   capture_ts_ns
   detection count
   selected target
   tracker state
   Kalman validity
   compensated target
   angular error
   controller output
   scheduler decision
   device execution result
   ```

33. Are runtime status payloads truthful about the active path?

   A status page that says DeepStream is available is not the same as saying
   DeepStream is selected and driving production detections.

## Product Shape

34. Is the host/UI tier outside the real-time critical path?

   The Jetson runtime should keep capture, inference, tracking, and control
   running if the browser or remote UI stalls.

35. Does the API expose small operational commands instead of requiring raw video
    transport for the control loop?

   Expected command shape:

   ```text
   load model
   activate model
   apply config revision
   start pipeline
   stop pipeline
   stream telemetry
   stream preview/debug
   report health
   ```

36. Are UI controls backed by the same `RuntimeConfig` fields used by runtime
    code, rather than disconnected display-only settings?

37. Are restart-required settings clearly separated from hot-reloadable settings?

## Anti-Patterns To Flag

Flag these as audit failures:

- A doc says DeepStream is the goal, but runtime config cannot select it.
- `gst-launch-1.0` succeeds, but production code still uses CPU appsink frames.
- DeepStream pipeline generation exists, but no live runtime owns the pipeline.
- TensorRT is used, but preprocessing still maps full frames to CPU.
- `DetectionBatch` exists, but downstream code still reads raw tensors or backend-specific objects.
- Runtime status says a backend is available, but not whether it is selected.
- Tests prove parser/config generation, but no Jetson evidence proves tensor meta extraction.
- Control output can be produced from stale bbox cache or pixel-domain error.
- Calibration changes do not clear tracker, controller memory, and queued commands.

## Suggested Final Audit Summary

Use this format when reporting:

```text
DeepStream active path: YES/PARTIAL/NO/UNKNOWN
NVMM/hardware decode continuity: YES/PARTIAL/NO/UNKNOWN
DetectionBatch as downstream seam: YES/PARTIAL/NO/UNKNOWN
Model manifest/config cache: YES/PARTIAL/NO/UNKNOWN
Angular calibrated control chain: YES/PARTIAL/NO/UNKNOWN
Scheduler/device safety: YES/PARTIAL/NO/UNKNOWN
Replay/log evidence: YES/PARTIAL/NO/UNKNOWN

Most important gap:
Smallest next verification:
Smallest next implementation slice:
```
