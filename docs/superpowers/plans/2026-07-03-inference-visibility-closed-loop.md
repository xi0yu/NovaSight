# Inference Visibility Closed Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make NovaSight show whether inference is truly running, expose detections to the frontend, draw real target boxes on the ROI preview, and improve TensorRT runtime diagnostics.

**Architecture:** Keep the current runtime service as the source of truth. Backend runtime state will include a compact detection list and selected target/control payload. Frontend dashboard will render those detections over the ROI preview using normalized CSS positioning derived from the runtime ROI size. TensorRT will report binding/output details and use TensorRT tensor dtypes when allocating output buffers.

**Tech Stack:** Python FastAPI runtime state, NovaSight inference contracts, TensorRT/cuda-python integration, React + TypeScript dashboard, existing CSS design system.

---

## Files

- Modify: `novasight/inference/contracts.py`
  - Add optional `debug` metadata on `InferenceResult`.
- Modify: `novasight/inference/onnxruntime_engine.py`
  - Fill `debug` with output shape and decoded detection count.
- Modify: `novasight/inference/tensorrt.py`
  - Use TensorRT output dtypes for host/device buffers.
  - Add output binding status to `status()`.
  - Fill `debug` with output shape and decoded detection count.
- Modify: `novasight/runtime/service.py`
  - Store mapped detection payloads in `vision`.
  - Store inference debug payload in `vision.inference`.
- Modify: `web/src/features/dashboard/DashboardView.tsx`
  - Read `vision.detections`.
  - Draw detection boxes and selected target marker over the ROI preview.
  - Show inference backend debug summary.
- Modify: `web/src/styles.css`
  - Add professional preview overlay styles.

## Task 1: Backend inference debug payload

- [x] Add `debug: dict[str, Any]` to `InferenceResult`.
- [x] In ONNX inference, include output shape and decoded count.
- [x] In TensorRT inference, include selected output name, shape, dtype, and decoded count.
- [x] Verify with `python3 -m compileall -q novasight`.

## Task 2: Runtime vision detection payload

- [x] Convert mapped detections into frontend-ready dictionaries.
- [x] Add `detections` to `_vision_status()`.
- [x] Add `debug` to `last_inference_status`.
- [x] Ensure empty and failed inference states clear stale detection boxes.
- [x] Verify with `python3 -m compileall -q novasight`.

## Task 3: Frontend ROI overlay

- [x] Parse `vision.detections` safely.
- [x] Draw each detection box based on ROI/input dimensions.
- [x] Highlight the selected target separately from ordinary detections.
- [x] Remove fake hard-coded FOV/target visual behavior from the preview.
- [x] Display backend inference debug summary in the inference result panel.
- [x] Verify with `pnpm --dir web build`.

## Task 4: TensorRT buffer correctness and diagnostics

- [x] Allocate each output buffer using `trt.nptype(engine.get_tensor_dtype(name))`.
- [x] Store host output buffers by output name instead of assuming one float32 output.
- [x] Cast selected output to float32 only before YOLO decoding.
- [x] Log TensorRT input/output bindings on load.
- [x] Verify local import/compile. Jetson runtime verification must be done on the target machine.

## Completion

- [x] Run `python3 -m compileall -q novasight`.
- [x] Run `pnpm --dir web build`.
- [x] Run `git diff --check`.
- [x] Commit the implementation.
