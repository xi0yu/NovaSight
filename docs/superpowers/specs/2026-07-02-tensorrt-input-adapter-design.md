# TensorRT Input Adapter Design

## Goal

Prepare NovaSight for a true GPU ROI to TensorRT path without pretending that CUDA bindings are already implemented in this repository.

## Design

Add a narrow TensorRT input adapter between `RoiFrame` and `TensorRtInferenceEngine`.

```text
RoiFrame
  -> TensorRT input adapter
    -> GPU buffer path when `roi_frame.gpu_buffer` exists
    -> CPU image fallback when only `roi_frame.image` exists
    -> unavailable result when neither exists
  -> TensorRtInferenceEngine
```

The adapter owns:

- Parsing model input shapes such as `1x3x640x640`.
- Reporting whether the current frame enters through `gpu_buffer` or `cpu_image`.
- Preserving ROI/source metadata for diagnostics.
- Avoiding fake CUDA execution in local tests.

The engine still returns empty detections until a real TensorRT execution context is wired in. This is intentional: this phase makes the input boundary explicit and testable.

## Runtime Behavior

- `TensorRtInferenceEngine.load()` validates and stores the parsed input shape.
- `TensorRtInferenceEngine.infer()` prepares an input buffer from the ROI frame.
- If `frame.gpu_buffer` exists, status records `last_input_mode="gpu_buffer"`.
- If only `frame.image` exists, status records `last_input_mode="cpu_image"`.
- If neither exists, inference returns `available=False` with a clear reason.

## Future CUDA Binding

The later CUDA implementation should replace the internals of the input adapter and engine execution only. `RuntimeService`, `RoiFrame`, capture ROI metadata, and detection coordinate mapping should remain stable.
