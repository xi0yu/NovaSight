# Capture Workbench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first real Jetson capture workbench: Chinese UI, full `/dev/video0` capability listing, front/back linked capture switching, and MJPEG preview.

**Architecture:** Keep the existing `CaptureService` as the camera boundary and add a small streaming layer above it. The frontend treats capture as a first-class workspace: capabilities, current state, apply action, and live preview all stay visible. Tests are deferred until the feature loop is complete, then added around broad critical paths instead of many narrow behavior tests.

**Tech Stack:** Python 3.10+, FastAPI, OpenCV, V4L2, React, TypeScript, Vite, CSS.

---

## File Structure

- Modify `novasight/capture/service.py`: add a safe single-frame read method for streaming without running a fixed-duration smoke loop.
- Modify `novasight/api/routes_capture.py`: add MJPEG stream route and any capture-switching response shape needed by the frontend.
- Modify `novasight/api/app.py`: keep route registration unchanged unless streaming needs shared state initialization.
- Modify `web/src/api.ts`: add capture capability types and API calls.
- Modify `web/src/App.tsx`: replace the English live/status console with a Chinese capture workbench view.
- Modify `web/src/styles.css`: update layout, preview, table, toolbar, status, and Chinese-friendly spacing.
- Modify `README.md`: add Jetson testing flow for capability switching and MJPEG preview.
- Modify or remove `tests/*` after implementation: clean obsolete narrow tests and add broad critical-path tests.

## Task 1: Backend Streaming Boundary

- [x] Add `CaptureService.read_frame()` that returns one `CapturedFrame | None`.
- [x] If `source` is not configured, keep the error explicit and diagnostic.
- [x] When a read returns `None`, increment dropped frame state and preserve the source for short transient failures.
- [x] When a read raises, mark capture unavailable using the existing recovery/close behavior.
- [x] Do not add narrow tests at this step; document behavior in code through simple names and reuse existing state fields.

## Task 2: MJPEG Stream Route

- [x] Add `GET /api/capture/stream.mjpg`.
- [x] On request, if no capture source exists, call `capture.configure(capture.config.device)`.
- [x] Stream multipart frames with `Content-Type: multipart/x-mixed-replace; boundary=frame`.
- [x] Encode frames with OpenCV `imencode(".jpg", frame.image)`.
- [x] If configuration fails, return HTTP 503 with the current capture state.
- [x] If frame read fails repeatedly, end the stream and leave `/api/capture/state` diagnostic.
- [x] Keep the endpoint compatible with `<img src="/api/capture/stream.mjpg">`.

## Task 3: Capture Capability API Shape

- [x] Keep `GET /api/capture/capabilities?device=/dev/video0` as the source of truth.
- [x] Preserve the existing parsed list: pixel format, width, height, fps list.
- [x] Confirm `POST /api/capture/select` accepts manual pixel format, width, height, and fps.
- [x] Ensure successful select swaps source only after the new source opens.
- [x] Ensure failed select keeps the last healthy source when one exists.

## Task 4: Frontend Capture API Client

- [x] Add TypeScript types for capture capabilities:
  - `CaptureCapability`
  - `CaptureCapabilitiesResponse`
  - `CaptureSelectPayload`
- [x] Add `getCaptureCapabilities(device: string)`.
- [x] Add `selectCaptureProfile(payload: CaptureSelectPayload)`.
- [x] Keep runtime state polling for current diagnostics.
- [x] Do not use old endpoints such as `/api/state`, `/api/dashboard`, or `/api/models/list`.

## Task 5: Chinese Capture Workbench UI

- [x] Replace English navigation labels:
  - Dashboard -> 总览
  - Live View -> 采集
  - Models -> 模型
  - Plugins -> 插件
  - Settings -> 设置
- [x] Make the capture page the main operator surface:
  - Device input: default `/dev/video0`
  - Refresh capabilities button: `刷新能力`
  - Preference buttons: `高帧率`、`低延迟`、`均衡`
  - Manual capability table with apply action: `应用`
  - Current profile panel
  - MJPEG preview panel
  - Diagnostics panel
- [x] Use `<img src="/api/capture/stream.mjpg">` for live preview.
- [x] Add a cache-busting query value after every successful profile switch so the browser reconnects the stream.
- [x] Show clear Chinese empty/error states:
  - `尚未读取采集能力`
  - `采集源未打开`
  - `切换失败，已保留上一组可用配置`

## Task 6: UI/UX Polish

- [x] Use a restrained industrial/workbench style, not a colorful landing page.
- [x] Keep cards only for repeated items and panels, avoid nested cards.
- [x] Make the capability table dense but readable.
- [x] Use tabular numbers for fps, latency, dropped frames, and frame period.
- [x] Ensure controls have at least 40px hit area.
- [x] Avoid `transition: all`; specify exact properties.
- [x] Keep all visible UI text Chinese.

## Task 7: Jetson Verification

- [ ] On Jetson, run:

```bash
python3 -m novasight doctor camera --device /dev/video0
```

- [ ] Start backend:

```bash
python3 -m novasight --host 0.0.0.0 --port 5174
```

- [ ] Verify API:

```bash
curl http://127.0.0.1:5174/api/capture/state
curl "http://127.0.0.1:5174/api/capture/capabilities?device=/dev/video0"
```

- [ ] Verify stream headers:

```bash
curl -I http://127.0.0.1:5174/api/capture/stream.mjpg
```

- [ ] Start frontend:

```bash
pnpm --dir web dev --host 0.0.0.0
```

- [ ] Open `http://<jetson-ip>:5173` and verify:
  - full capability list appears
  - applying `MJPG 1920x1080@240` reconnects preview
  - applying another supported profile updates state
  - failure preserves previous healthy stream

## Task 8: Consolidated Tests After Feature Completion

- [x] Remove obsolete tests that assert deleted English UI or old API paths.
- [x] Add broad backend tests for:
  - capability parsing
  - manual profile selection
  - select failure preserving healthy capture state
  - MJPEG endpoint returning multipart response when a fake frame source is available
- [x] Add broad frontend type/build checks:

```bash
pnpm --dir web typecheck
pnpm --dir web build
```

- [x] Run Python checks:

```bash
pytest -q
```

- [x] Accept around 80% meaningful coverage; do not add tests solely to chase 100%.

## Self-Review

- Spec coverage: The plan covers full capability listing, linked capture switching, MJPEG preview, Chinese UI, Jetson verification, and delayed broad tests.
- Placeholder scan: No `TBD` or open-ended implementation placeholders remain.
- Type consistency: Capture profile, capability, and stream names match current project terminology.
