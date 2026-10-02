# Capture Session Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace preview-owned and runtime-owned capture reads with a single automatically started capture session that owns the device and feeds preview plus inference/control consumers.

**Architecture:** Add `CaptureSession` as the only component allowed to read from `FrameSource`. Keep `CaptureService` as the API-facing coordinator for capability selection and session lifecycle. Make MJPEG preview and `RuntimePipeline` consume `latest_frame()` from the session, never directly opening or reading the device.

**Tech Stack:** Python `threading`, FastAPI, existing `FrameSource`/GStreamer appsink capture stack, React + TypeScript UI, pytest.

---

## File Structure

- Create `novasight/capture/session.py`
  - Owns `FrameSource`, capture thread, latest-frame cache, capture metrics, start/stop/reconfigure lifecycle.
- Modify `novasight/capture/service.py`
  - Removes direct source ownership/read loops from API-facing service.
  - Delegates active capture work to `CaptureSession`.
  - Keeps profile selection, config mutation, capability lookup, preview metric helpers.
- Modify `novasight/api/routes_capture.py`
  - `/api/capture/select` applies profile and starts capture.
  - `/api/capture/stream.mjpg` becomes a pure latest-frame consumer.
  - Adds `POST /api/capture/stop` for explicit capture stop.
- Modify `novasight/runtime/pipeline.py`
  - Stops opening/reading capture source.
  - Starts inference/control only if capture session is running.
  - Consumes frames via `latest_frame(after_frame_id=...)`.
- Modify `novasight/api/routes_runtime.py`
  - Runtime start error should say capture must be running.
  - Stop runtime should stop inference/control only, not capture.
- Modify `web/src/api.ts`
  - Add `captureStop`.
  - Rename runtime start/stop exports to inference/control semantics if needed.
- Modify `web/src/App.tsx`
  - Remove `启动主线` and `停止主线`.
  - Capture page actions become `应用并启动采集` and `停止采集`.
  - Runtime controls become `启动推理控制` and `停止推理控制`.
  - Preview empty state is product-facing Chinese, not internal fallback wording.
- Modify `web/src/styles.css`
  - Keep button styling, adjust class names only if needed.
- Tests:
  - `tests/test_capture_session.py`
  - `tests/test_capture_service.py`
  - `tests/test_capture_api.py`
  - `tests/test_runtime_pipeline.py`
  - Existing frontend type/build checks.

---

### Task 1: Add CaptureSession as the Single FrameSource Owner

**Files:**
- Create: `novasight/capture/session.py`
- Test: `tests/test_capture_session.py`

- [ ] **Step 1: Write failing tests for session lifecycle and latest frame publishing**

Create `tests/test_capture_session.py` with:

```python
import time

from novasight.capture.session import CaptureSession
from novasight.capture.source import CapturedFrame
from novasight.capture.state import CaptureProfile


class CountingSource:
    backend_label = "gst-appsink:test"

    def __init__(self) -> None:
        self.count = 0
        self.closed = False

    def read(self):
        self.count += 1
        return CapturedFrame(
            frame_id=self.count,
            width=320,
            height=320,
            pixel_format="NV12",
            ts_ns=time.monotonic_ns(),
            capture_wait_ms=0.1,
            image=None,
        )

    def close(self) -> None:
        self.closed = True


def _profile() -> CaptureProfile:
    return CaptureProfile(
        device="/dev/video0",
        pixel_format="MJPG",
        width=1920,
        height=1080,
        fps=120,
        preference="auto_high_fps",
        selection_reason="test",
    )


def test_capture_session_starts_thread_and_publishes_latest_frame() -> None:
    source = CountingSource()
    session = CaptureSession(source_factory=lambda profile: source)

    state = session.start(_profile())
    frame = session.latest_frame(after_frame_id=0, timeout_s=0.2)
    stopped = session.stop("test complete")

    assert state.available is True
    assert frame is not None
    assert frame.frame_id >= 1
    assert session.running is False
    assert stopped.available is False
    assert source.closed is True


def test_capture_session_reconfigure_closes_previous_source() -> None:
    sources: list[CountingSource] = []

    def factory(profile):
        source = CountingSource()
        sources.append(source)
        return source

    session = CaptureSession(source_factory=factory)
    session.start(_profile())
    session.reconfigure(_profile())
    session.stop("done")

    assert len(sources) == 2
    assert sources[0].closed is True
    assert sources[1].closed is True


def test_capture_session_latest_frame_returns_only_newer_frames() -> None:
    source = CountingSource()
    session = CaptureSession(source_factory=lambda profile: source)
    session.start(_profile())
    first = session.latest_frame(after_frame_id=0, timeout_s=0.2)
    assert first is not None

    stale = session.latest_frame(after_frame_id=first.frame_id, timeout_s=0)
    session.stop("done")

    assert stale is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_capture_session.py -q
```

Expected: FAIL because `novasight.capture.session` does not exist.

- [ ] **Step 3: Implement `CaptureSession`**

Create `novasight/capture/session.py`:

```python
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from .source import CapturedFrame, FrameSource
from .state import CaptureProfile, CaptureRuntimeState

logger = logging.getLogger("novasight.capture.session")


class CaptureSession:
    def __init__(
        self,
        *,
        source_factory: Callable[[CaptureProfile], FrameSource],
        empty_read_sleep_s: float = 0.001,
    ) -> None:
        self.source_factory = source_factory
        self.empty_read_sleep_s = empty_read_sleep_s
        self.state = CaptureRuntimeState()
        self._source: FrameSource | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._latest_frame: CapturedFrame | None = None
        self._last_frame_ts_ns: int | None = None

    @property
    def running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    @property
    def source(self) -> FrameSource | None:
        with self._lock:
            return self._source

    def start(self, profile: CaptureProfile) -> CaptureRuntimeState:
        with self._lock:
            if self.running and self.state.profile == profile:
                return self.state
            self._stop_unlocked("restarting capture")
            source = self.source_factory(profile)
            self._source = source
            self._latest_frame = None
            self._last_frame_ts_ns = None
            self._stop.clear()
            self.state = CaptureRuntimeState(
                available=True,
                device=profile.device,
                profile=profile,
                backend=source.backend_label,
                last_error=None,
            )
            self._thread = threading.Thread(
                target=self._run_loop,
                name="novasight-capture-session",
                daemon=True,
            )
            self._thread.start()
            self._condition.notify_all()
            return self.state

    def reconfigure(self, profile: CaptureProfile) -> CaptureRuntimeState:
        return self.start(profile)

    def stop(self, reason: str | None = None) -> CaptureRuntimeState:
        with self._lock:
            self._stop_unlocked(reason or "capture stopped")
            return self.state

    def latest_frame(
        self,
        *,
        after_frame_id: int | None = None,
        timeout_s: float = 0.0,
    ) -> CapturedFrame | None:
        deadline = time.monotonic() + max(timeout_s, 0.0)
        with self._condition:
            while True:
                frame = self._latest_frame
                if frame is not None and (
                    after_frame_id is None or frame.frame_id > after_frame_id
                ):
                    return frame
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                source = self._source
            if source is None:
                return
            try:
                frame = source.read()
            except Exception as exc:
                self._mark_unavailable(f"capture read failed: {exc}")
                return
            if frame is None:
                with self._lock:
                    self.state.frames_dropped += 1
                if self.empty_read_sleep_s > 0:
                    time.sleep(self.empty_read_sleep_s)
                continue
            self._publish_frame(frame)

    def _publish_frame(self, frame: CapturedFrame) -> None:
        with self._condition:
            self.state.available = True
            self.state.capture_wait_ms = frame.capture_wait_ms
            if self._last_frame_ts_ns is not None:
                self.state.frame_period_ms = (frame.ts_ns - self._last_frame_ts_ns) / 1e6
                if self.state.frame_period_ms > 0:
                    self.state.fps_capture = 1000.0 / self.state.frame_period_ms
            self._last_frame_ts_ns = frame.ts_ns
            self._latest_frame = frame
            self.state.preview_frames += 1
            self.state.last_error = None
            self._condition.notify_all()

    def _mark_unavailable(self, reason: str) -> None:
        with self._condition:
            self._close_source_unlocked()
            self.state.available = False
            self.state.last_error = reason
            self._latest_frame = None
            self._last_frame_ts_ns = None
            self._condition.notify_all()

    def _stop_unlocked(self, reason: str) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            self._lock.release()
            try:
                thread.join(timeout=1.0)
            finally:
                self._lock.acquire()
        self._thread = None
        close_error = self._close_source_unlocked()
        self.state.available = False
        self.state.last_error = close_error or reason
        self._latest_frame = None
        self._last_frame_ts_ns = None
        self._condition.notify_all()

    def _close_source_unlocked(self) -> str:
        source = self._source
        self._source = None
        if source is None:
            return ""
        try:
            source.close()
        except Exception as exc:
            return f"capture close failed: {exc}"
        return ""
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
pytest tests/test_capture_session.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add novasight/capture/session.py tests/test_capture_session.py
git commit -m "feat: add capture session owner"
```

---

### Task 2: Make CaptureService Coordinate the Session

**Files:**
- Modify: `novasight/capture/service.py`
- Test: `tests/test_capture_service.py`

- [ ] **Step 1: Write failing service tests**

Add to `tests/test_capture_service.py`:

```python
def test_configure_starts_capture_session_and_publishes_frames() -> None:
    source = FakeSource()
    service = _service(source_factory=lambda profile: source)

    state = service.configure("/dev/video0")
    frame = service.wait_preview_frame(after_frame_id=0, timeout_s=0.2)
    service.stop("test complete")

    assert state.available is True
    assert frame is not None
    assert source.count >= 1
    assert source.closed is True


def test_service_exposes_session_running_state_after_configure() -> None:
    service = _service()

    state = service.configure("/dev/video0")

    assert state.available is True
    assert service.source is not None
    assert service.session.running is True

    service.stop("test complete")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_capture_service.py::test_configure_starts_capture_session_and_publishes_frames tests/test_capture_service.py::test_service_exposes_session_running_state_after_configure -q
```

Expected: FAIL because `configure()` currently only opens source instead of starting a capture thread.

- [ ] **Step 3: Refactor `CaptureService` to own a `CaptureSession`**

Modify `novasight/capture/service.py`:

- Import `CaptureSession`.
- Replace direct `self.source`, `_latest_preview_frame`, and direct read helpers with `self.session`.
- Keep `source` property for compatibility as read-only proxy to `self.session.source`.
- Make `configure()` select a profile and call `self.session.reconfigure(profile)`.
- Keep legacy `read_frame()` temporarily only if existing callers still need it during this task; it must be removed after Task 4 when preview and runtime have migrated.
- Add `stop(reason: str | None = None)`.
- Make `wait_preview_frame()` delegate to `session.latest_frame()`.
- Keep `record_preview_output()` and `record_preview_drop()` updating `self.state`.

Use this shape:

```python
from .session import CaptureSession

class CaptureService:
    def __init__(...):
        ...
        self.session = CaptureSession(
            source_factory=self.source_factory,
            empty_read_sleep_s=empty_read_sleep_s,
        )
        self.state = self.session.state

    @property
    def source(self):
        return self.session.source

    def _sync_state(self) -> CaptureRuntimeState:
        self.state = self.session.state
        return self.state

    def configure(...):
        with self._source_lock:
            return self._configure_unlocked(...)

    def _configure_unlocked(...):
        ...
        try:
            profile = select_capture_profile(...)
            state = self.session.reconfigure(profile)
        except Exception as exc:
            failure = CaptureRuntimeState(...)
            self.last_config_error = failure
            self.state = failure if self.session.source is None else self.session.state
            return self.state
        self.config.device = selected_device
        ...
        self.last_config_error = None
        return self._sync_state()

    def stop(self, reason: str | None = None) -> CaptureRuntimeState:
        state = self.session.stop(reason or "capture stopped")
        return self._sync_state()

    def wait_preview_frame(...):
        return self.session.latest_frame(after_frame_id=after_frame_id, timeout_s=timeout_s)
```

- [ ] **Step 4: Update or remove old service tests that call direct reads**

Replace tests that used `read_frame()` with `wait_preview_frame()` after `configure()`:

```python
frame = service.wait_preview_frame(after_frame_id=0, timeout_s=0.2)
```

For diagnostics assertions, use the state after the session has produced at least two frames:

```python
first = service.wait_preview_frame(after_frame_id=0, timeout_s=0.2)
assert first is not None
second = service.wait_preview_frame(after_frame_id=first.frame_id, timeout_s=0.2)
assert second is not None
assert service.state.fps_capture > 0
```

- [ ] **Step 5: Run service tests**

Run:

```bash
pytest tests/test_capture_service.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add novasight/capture/service.py tests/test_capture_service.py
git commit -m "refactor: route capture service through session"
```

---

### Task 3: Make MJPEG Preview a Pure Session Consumer

**Files:**
- Modify: `novasight/api/routes_capture.py`
- Test: `tests/test_capture_api.py`

- [ ] **Step 1: Write failing API tests**

Modify or add tests in `tests/test_capture_api.py`:

```python
def test_capture_stream_returns_503_when_capture_session_not_running(tmp_path) -> None:
    app = create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml")
    cfg = RuntimeConfig()
    source = CachedPreviewSource()
    service = CaptureService(
        config=cfg.capture,
        capability_runner=lambda device: CAPS_TEXT,
        source_factory=lambda profile: source,
    )
    app.state.capture = service
    client = _client(app)

    response = client.get("/api/capture/stream.mjpg")

    assert response.status_code == 503
    assert "采集未启动" in response.text
    assert source.count == 0


def test_capture_select_starts_session_for_preview(tmp_path) -> None:
    app = create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml")
    cfg = RuntimeConfig()
    source = CachedPreviewSource()
    service = CaptureService(
        config=cfg.capture,
        capability_runner=lambda device: CAPS_TEXT,
        source_factory=lambda profile: source,
    )
    app.state.capture = service
    client = _client(app)

    select = client.post("/api/capture/select", json={"device": "/dev/video0"})
    response = client.get("/api/capture/stream.mjpg")
    service.stop("test complete")

    assert select.status_code == 200
    assert response.status_code == 200
    assert b"Content-Type: image/jpeg" in response.content
    assert source.count >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_capture_api.py::test_capture_stream_returns_503_when_capture_session_not_running tests/test_capture_api.py::test_capture_select_starts_session_for_preview -q
```

Expected: FAIL because stream currently configures/reads when not running.

- [ ] **Step 3: Update capture routes**

Modify `novasight/api/routes_capture.py`:

- In `stream()`, do not call `capture.configure()`.
- If `capture.state.available is False` or `capture.source is None`, return `503` with state plus `last_error="采集未启动，无法打开预览。"` unless there is a more specific error.
- In `_mjpeg_frames()`, remove `capture.read_frame()` and `_runtime_is_running()`.
- Always call `capture.wait_preview_frame(after_frame_id=last_frame_id, timeout_s=interval_s)`.
- On timeout, record preview drop and continue.
- Add `POST /api/capture/stop`:

```python
@router.post("/stop")
def stop(request: Request) -> dict:
    return asdict(request.app.state.capture.stop("capture stopped by user"))
```

- [ ] **Step 4: Run capture API tests**

Run:

```bash
pytest tests/test_capture_api.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add novasight/api/routes_capture.py tests/test_capture_api.py
git commit -m "fix: make preview consume capture session only"
```

---

### Task 4: Make RuntimePipeline Consume Existing Capture Session

**Files:**
- Modify: `novasight/runtime/pipeline.py`
- Modify: `novasight/api/routes_runtime.py`
- Test: `tests/test_runtime_pipeline.py`
- Test: `tests/test_runtime_api.py`

- [ ] **Step 1: Write failing runtime pipeline tests**

In `tests/test_runtime_pipeline.py`, add:

```python
def test_runtime_pipeline_requires_running_capture_session() -> None:
    capture = FakeCapture()
    runtime = FakeRuntime()
    pipeline = RuntimePipeline(capture=capture, runtime=runtime)

    try:
        pipeline.start()
    except RuntimeError as exc:
        assert "采集未启动" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")

    assert capture.configure_calls == []


def test_runtime_pipeline_consumes_latest_frames_without_reading_source() -> None:
    capture = FakeCapture(running=True)
    runtime = FakeRuntime()
    pipeline = RuntimePipeline(capture=capture, runtime=runtime)

    pipeline.start()
    deadline = time.monotonic() + 0.5
    while not runtime.frames and time.monotonic() < deadline:
        time.sleep(0.01)
    pipeline.stop()

    assert runtime.frames
    assert capture.read_calls == 0
```

If current test helpers do not have these fields, define local fakes:

```python
class FakeCapture:
    def __init__(self, running: bool = False) -> None:
        self.state = SimpleNamespace(available=running)
        self.configure_calls = []
        self.read_calls = 0
        self._frame_id = 0

    def latest_frame(self, *, after_frame_id=None, timeout_s=0.0):
        if not self.state.available:
            return None
        self._frame_id += 1
        return CapturedFrame(
            frame_id=self._frame_id,
            width=320,
            height=320,
            pixel_format="NV12",
            ts_ns=time.monotonic_ns(),
            capture_wait_ms=0.1,
            image=None,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_runtime_pipeline.py -q
```

Expected: FAIL because runtime currently configures and reads capture.

- [ ] **Step 3: Update `RuntimePipeline`**

Modify `novasight/runtime/pipeline.py`:

- Remove capture thread from `_threads`.
- `start()` must not call `capture.configure()`.
- `start()` checks `capture.state.available`; if false, raise `RuntimeError("采集未启动，无法启动推理控制。")`.
- Only start one thread named `novasight-inference-control`.
- That thread polls `capture.latest_frame(after_frame_id=last_frame_id, timeout_s=0.1)`.
- It puts/processes latest frames without owning source reads.

Core loop:

```python
def _runtime_loop(self) -> None:
    last_frame_id = 0
    while not self._stop.is_set():
        frame = self.capture.latest_frame(
            after_frame_id=last_frame_id,
            timeout_s=0.1,
        )
        if frame is None:
            continue
        last_frame_id = frame.frame_id
        self.runtime.process_captured_frame(frame)
        self.stats.processed_frames += 1
        self.stats.last_frame_id = frame.frame_id
```

- [ ] **Step 4: Update runtime API tests if needed**

In `tests/test_runtime_api.py`, ensure `POST /api/runtime/start` returns a clear failure if capture is not running:

```python
def test_runtime_start_requires_capture_session(tmp_path) -> None:
    app = create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml")
    client = _client(app)

    response = client.post("/api/runtime/start")

    assert response.status_code == 400
    assert "采集未启动" in response.text
```

Update `routes_runtime.start_runtime()` to convert this `RuntimeError` into `HTTPException(status_code=400, detail=str(exc))`.

- [ ] **Step 5: Run runtime tests**

Run:

```bash
pytest tests/test_runtime_pipeline.py tests/test_runtime_api.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add novasight/runtime/pipeline.py novasight/api/routes_runtime.py tests/test_runtime_pipeline.py tests/test_runtime_api.py
git commit -m "refactor: run inference from capture session"
```

---

### Task 5: Rename and Reshape Frontend Controls Around Product Actions

**Files:**
- Modify: `web/src/api.ts`
- Modify: `web/src/App.tsx`
- Modify: `web/src/styles.css`

- [ ] **Step 1: Update API client**

In `web/src/api.ts`:

```ts
export const API_PATHS = {
  ...
  captureStop: "/api/capture/stop",
  inferenceStart: "/api/runtime/start",
  inferenceStop: "/api/runtime/stop",
  ...
} as const;

export function stopCapture(): Promise<CaptureState> {
  return requestJson<CaptureState>(API_PATHS.captureStop, {
    method: "POST"
  });
}

export function startInferenceControl(): Promise<Record<string, unknown>> {
  return requestJson<Record<string, unknown>>(API_PATHS.inferenceStart, {
    method: "POST"
  });
}

export function stopInferenceControl(): Promise<Record<string, unknown>> {
  return requestJson<Record<string, unknown>>(API_PATHS.inferenceStop, {
    method: "POST"
  });
}
```

Remove or stop using `startRuntime()` / `stopRuntime()` in UI code.

- [ ] **Step 2: Update UI wording**

In `web/src/App.tsx`:

- Replace `RuntimeControl` with `InferenceControl`.
- Capture workbench action area:
  - `应用并启动采集` on profile/config apply buttons.
  - `停止采集` button when `capture?.available` is true.
- Remove all user-facing `主线`, `兜底`, `fallback`, `缓存消费者`.
- Add product-facing note:

```tsx
<div className="inline-note">
  浏览器预览低优先级，不影响采集与推理。
</div>
```

- Runtime/inference buttons use:
  - `启动推理控制`
  - `停止推理控制`

- [ ] **Step 3: Run text scan**

Run:

```bash
rg -n "主线|兜底|fallback|缓存消费者|启动 Runtime|运行链路" web/src
```

Expected: no matches, except comments if intentionally explaining removed terminology. Prefer no matches.

- [ ] **Step 4: Run frontend checks**

Run:

```bash
pnpm --dir web typecheck
pnpm --dir web build
```

Expected: both pass.

- [ ] **Step 5: Commit**

```bash
git add web/src/api.ts web/src/App.tsx web/src/styles.css
git commit -m "fix: align capture ui with session lifecycle"
```

---

### Task 5.5: Remove Legacy Direct Read Methods

**Files:**
- Modify: `novasight/capture/service.py`
- Test: `tests/test_capture_service.py`

- [ ] **Step 1: Write failing test that direct read methods are gone**

Add to `tests/test_capture_service.py`:

```python
def test_capture_service_has_no_direct_source_read_api() -> None:
    service = _service()

    assert not hasattr(service, "read_frame")
    assert not hasattr(service, "capture_frames")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_capture_service.py::test_capture_service_has_no_direct_source_read_api -q
```

Expected: FAIL because the legacy methods still exist.

- [ ] **Step 3: Remove legacy direct read methods**

In `novasight/capture/service.py`, delete:

- `read_frame`
- `capture_frames`
- `_recover_source` if it is now only used by `capture_frames`
- `_mark_unavailable` / `mark_unavailable` if no route or pipeline uses them

Keep explicit session lifecycle methods:

```python
def stop(self, reason: str | None = None) -> CaptureRuntimeState:
    state = self.session.stop(reason or "capture stopped")
    return self._sync_state()
```

If tests still need unavailable state injection, update tests to use `service.stop("reason")` or `service.session.stop("reason")`.

- [ ] **Step 4: Run service, API, and runtime tests**

Run:

```bash
pytest tests/test_capture_service.py tests/test_capture_api.py tests/test_runtime_pipeline.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add novasight/capture/service.py tests/test_capture_service.py tests/test_capture_api.py tests/test_runtime_pipeline.py
git commit -m "refactor: remove direct capture read api"
```

---

### Task 6: Remove Slow Production Fallbacks From Jetson Capture Path

**Files:**
- Modify: `novasight/capture/pipeline.py`
- Modify: `novasight/capture/service.py`
- Test: `tests/test_capture_pipeline.py`
- Test: `tests/test_capture_service.py`

- [ ] **Step 1: Write failing tests for no OpenCV fallback in production default**

In `tests/test_capture_service.py`, add:

```python
def test_default_source_factory_reports_appsink_failures_without_opencv_fallback(monkeypatch) -> None:
    from novasight.capture import service as service_module

    class FailingAppSinkSource:
        def __init__(self, profile, candidate) -> None:
            raise RuntimeError("appsink unavailable")

    class FailingOpenCvSource:
        def __init__(self, profile, candidate) -> None:
            raise AssertionError("OpenCV fallback should not run for Jetson production capture")

    monkeypatch.setattr(service_module, "GstAppSinkFrameSource", FailingAppSinkSource)
    monkeypatch.setattr(service_module, "OpenCvFrameSource", FailingOpenCvSource)

    profile = service_module.select_capture_profile(
        "/dev/video0",
        service_module.query_capabilities("/dev/video0", runner=lambda device: CAPS_TEXT).capabilities,
    )

    try:
        service_module._open_default_source(profile)
    except RuntimeError as exc:
        assert "appsink unavailable" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_capture_service.py::test_default_source_factory_reports_appsink_failures_without_opencv_fallback -q
```

Expected: FAIL because `_open_default_source()` currently tries OpenCV after appsink failure.

- [ ] **Step 3: Remove OpenCV fallback from default source factory**

Modify `_open_default_source()` in `novasight/capture/service.py`:

```python
def _open_default_source(profile: CaptureProfile) -> FrameSource:
    return _open_first_readable_source(
        profile,
        candidates=build_appsink_candidates(profile),
        source_cls=GstAppSinkFrameSource,
    )
```

Keep `OpenCvFrameSource` code only for explicit future diagnostics/tests if existing tests need it. It must not be part of default Jetson production path.

- [ ] **Step 4: Run capture pipeline and service tests**

Run:

```bash
pytest tests/test_capture_pipeline.py tests/test_capture_service.py -q
```

Expected: PASS after updating tests that expected OpenCV fallback.

- [ ] **Step 5: Commit**

```bash
git add novasight/capture/service.py tests/test_capture_service.py tests/test_capture_pipeline.py
git commit -m "fix: fail explicitly for missing jetson appsink capture"
```

---

### Task 7: Final Integration Verification

**Files:**
- No required source changes unless verification exposes a bug.

- [ ] **Step 1: Run backend tests**

Run:

```bash
pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Run frontend checks**

Run:

```bash
pnpm --dir web typecheck
pnpm --dir web build
```

Expected: both pass.

- [ ] **Step 3: Run internal wording scan**

Run:

```bash
rg -n "主线|兜底|fallback|缓存消费者" web/src novasight docs/superpowers/specs/2026-07-02-capture-session-design.md
```

Expected:

- `web/src` and `novasight` have no user-facing matches.
- The design doc may mention the removed terms as anti-requirements.

- [ ] **Step 4: Run diff check**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 5: Commit any verification fixes**

Only if Step 1-4 required code fixes:

```bash
git add <changed-files>
git commit -m "fix: stabilize capture session integration"
```

- [ ] **Step 6: Push develop-alpha**

Run:

```bash
git push origin develop-alpha
```

Expected: remote `develop-alpha` updated.

---

## Manual Jetson Test Procedure

Run after implementation reaches Jetson:

1. Start backend:

```bash
python -m novasight --host 0.0.0.0 --port 5174
```

2. Open frontend.
3. Apply `MJPG 1920x1080@120` with `应用并启动采集`.
4. Confirm UI state is `采集中`.
5. Confirm backend label is a `gst-appsink:nvmm-mjpg-*` route.
6. Confirm `采集帧率` reflects capture session performance, not preview fps.
7. Set preview to `15` or `30`.
8. Confirm preview open/close does not materially lower capture fps.
9. Start `推理控制`.
10. Confirm capture remains running independently.

---

## Self-Review

- Spec coverage:
  - Single capture session: Task 1 and Task 2.
  - Preview pure consumer: Task 3.
  - Runtime consumes capture session: Task 4.
  - UI removes internal terms and productizes controls: Task 5.
  - Explicit Jetson path without slow fallback: Task 6.
  - Verification and Jetson manual procedure: Task 7 plus manual test procedure.
- Placeholder scan:
  - No placeholder markers or unspecified "add tests" steps.
- Type consistency:
  - Plan consistently uses `CaptureSession.latest_frame()`, `CaptureService.wait_preview_frame()`, and `CaptureState`.
