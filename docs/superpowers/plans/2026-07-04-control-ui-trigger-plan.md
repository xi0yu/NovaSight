# Control UI Trigger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make control telemetry readable, make inference overlays visually distinct, and expose the trigger mode used by the aim/control pipeline.

**Architecture:** Keep control calculation continuous for telemetry, while output permission remains runtime-gated. Add a small `control.trigger_mode` config field and use it in `RuntimeService` to decide whether kmNet requires hardware input. Update Studio UI and CSS without changing the model/capture pipeline.

**Tech Stack:** Python dataclass config, FastAPI runtime config update path, React/Vite/TypeScript, CSS.

---

### Task 1: Trigger Mode Config

**Files:**
- Modify: `novasight/config/runtime.py`
- Modify: `novasight/config/schema.py`
- Modify: `config/novasight.example.yaml`
- Modify: `novasight/runtime/service.py`

- [x] Add `trigger_mode: str = "hardware"` to `ControlConfig`.
- [x] Add schema metadata for `control.trigger_mode` with choices `hardware`, `telemetry`, `always`.
- [x] Add example YAML value.
- [x] In runtime, require hardware trigger only when `output_mode == "kmnet"`, hardware is active, and `trigger_mode != "always"`.

### Task 2: Studio Controls

**Files:**
- Modify: `web/src/features/studio/StudioConsoleView.tsx`

- [x] Read `controlConfig.trigger_mode`.
- [x] Add a select in 参数设置 / 鼠标移动算法:
  - `hardware`: 硬件按键触发
  - `telemetry`: 持续计算，按键发送
  - `always`: 调试直出
- [x] Show trigger mode in metrics and control feedback.

### Task 3: Overlay And Feedback Styling

**Files:**
- Modify: `web/src/features/studio/StudioConsoleView.tsx`
- Modify: `web/src/styles.css`

- [x] Add class-based detection overlay colors by class id.
- [x] Keep selected target visually distinct from normal detections.
- [x] Make control feedback use compact rows with stable labels and readable values.
- [x] Make long execution messages span full width.

### Task 4: Verification

**Commands:**
- `python3 -m compileall -q novasight`
- `pnpm --dir web build`
- `git diff --check`

- [x] Commit and push to `origin develop-alpha`.
