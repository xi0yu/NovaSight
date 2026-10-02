# TensorRT Input Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a testable TensorRT input adapter that chooses GPU-buffer input when available and CPU-image fallback otherwise.

**Architecture:** A new `novasight/inference/input.py` module parses model input shapes and prepares a small input descriptor from a `RoiFrame`. `TensorRtInferenceEngine` stores the loaded input shape, calls the adapter during infer, and reports the selected input mode in status.

**Tech Stack:** Python dataclasses, existing inference contracts, pytest.

---

## Files

- Create: `novasight/inference/input.py`
- Modify: `novasight/inference/tensorrt.py`
- Modify: `novasight/inference/__init__.py`
- Test: `tests/test_inference_runtime.py`

## Tasks

- [x] Add failing tests for input shape parsing and TensorRT GPU/CPU input selection.
- [x] Run focused tests and confirm failures.
- [x] Implement `TensorInputShape`, `PreparedTensorInput`, and `prepare_tensor_input()`.
- [x] Wire `TensorRtInferenceEngine.load()`, `infer()`, and `status()` through the adapter.
- [x] Run focused tests and confirm pass.
- [ ] Run full backend and frontend verification.
