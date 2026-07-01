# NovaSight Phase 2 Capture, Inference, and Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Phase 2 vertical slice: Jetson-first `/dev/video0` capture discovery and smoke testing, TensorRT inference boundary, one experimental vision/control plugin pair, and bounded silent/console control output for future kmNet.

**Architecture:** Add focused `capture`, `inference`, and `control` packages without moving Phase 1 registry/plugin/API ownership. Runtime state grows capture and inference sections; plugins still receive `FrameContext`; executors consume bounded control output rather than raw hardware calls.

**Tech Stack:** Python 3.10+, FastAPI, SQLite standard library, pytest, OpenCV/GStreamer through optional `cv2`, V4L2 via `v4l2-ctl`, optional TensorRT imported lazily, React/Vite/TypeScript.

---

## File Structure

Create:

- `novasight/capture/__init__.py`: capture package exports.
- `novasight/capture/state.py`: capture dataclasses.
- `novasight/capture/caps.py`: `v4l2-ctl` runner and parser.
- `novasight/capture/profile.py`: automatic/manual profile selection.
- `novasight/capture/pipeline.py`: GStreamer/OpenCV candidate labels and pipeline strings.
- `novasight/capture/source.py`: source protocol, captured frame, OpenCV/GStreamer adapter.
- `novasight/capture/service.py`: capture state service and smoke loop.
- `novasight/api/routes_capture.py`: capture capability/state/select endpoints.
- `novasight/inference/__init__.py`: inference package exports.
- `novasight/inference/contracts.py`: inference result and engine protocol.
- `novasight/inference/unavailable.py`: unavailable/fake-safe engine.
- `novasight/inference/tensorrt.py`: lazy TensorRT engine wrapper.
- `novasight/inference/runtime.py`: active model to inference engine resolver.
- `novasight/control/__init__.py`: control package exports.
- `novasight/control/output.py`: `ControlOutput`, policy, silent/console modes.
- `tests/test_capture_caps.py`
- `tests/test_capture_profile.py`
- `tests/test_capture_pipeline.py`
- `tests/test_capture_service.py`
- `tests/test_capture_api.py`
- `tests/test_inference_runtime.py`
- `tests/test_experimental_plugins.py`
- `tests/test_control_output.py`

Modify:

- `novasight/config/runtime.py`: add `CaptureConfig` and `ControlConfig`.
- `config/novasight.example.yaml`: document capture and control runtime fields.
- `novasight/main.py`: add `doctor camera` and `capture-smoke` subcommands.
- `novasight/api/app.py`: create and attach capture/inference services; include capture router.
- `novasight/runtime/state.py`: add capture/inference state fields.
- `novasight/runtime/service.py`: add captured-frame inference flow while preserving `process_frame`.
- `novasight/plugins/contracts.py`: keep compatibility, no hardware types.
- `novasight/plugins/builtin.py`: add experimental vision/control plugins.
- `novasight/plugins/runtime.py`: register experimental plugins.
- `novasight/executors/contracts.py`: accept bounded output or keep raw compatibility through adapter.
- `novasight/executors/dry_run.py`: record bounded output policy result.
- `novasight/executors/runtime.py`: route intents through `ControlOutputPolicy`.
- `novasight/api/routes_health.py`: return expanded runtime state.
- `web/src/api.ts`: add capture state types.
- `web/src/App.tsx`: show capture status panel.
- `web/src/styles.css`: small table/status styling only.
- `README.md`: Phase 2 Jetson diagnostics.

---

### Task 1: V4L2 Capability Parser

**Files:**
- Create: `novasight/capture/__init__.py`
- Create: `novasight/capture/state.py`
- Create: `novasight/capture/caps.py`
- Test: `tests/test_capture_caps.py`

- [ ] **Step 1: Write failing parser tests**

Create `tests/test_capture_caps.py`:

```python
from novasight.capture import CaptureCapability, query_capabilities
from novasight.capture.caps import parse_v4l2_formats


V4L2_TEXT = """
[0]: 'MJPG' (Motion-JPEG, compressed)
    Size: Discrete 1920x1080
        Interval: Discrete 0.007s (144.000 fps)
        Interval: Discrete 0.008s (120.000 fps)
        Interval: Discrete 0.017s (60.000 fps)
    Size: Discrete 1280x720
        Interval: Discrete 0.007s (144.000 fps)
[1]: 'NV12' (Y/CbCr 4:2:0)
    Size: Discrete 1920x1080
        Interval: Discrete 0.017s (60.000 fps)
[2]: 'YUYV' (YUYV 4:2:2)
    Size: Discrete 1280x720
        Interval: Discrete 0.033s (30.000 fps)
"""


def test_parse_v4l2_formats_groups_format_size_and_fps() -> None:
    caps = parse_v4l2_formats(V4L2_TEXT)

    assert caps == [
        CaptureCapability("MJPG", 1920, 1080, [144, 120, 60]),
        CaptureCapability("MJPG", 1280, 720, [144]),
        CaptureCapability("NV12", 1920, 1080, [60]),
        CaptureCapability("YUYV", 1280, 720, [30]),
    ]


def test_query_capabilities_reports_unavailable_when_runner_fails() -> None:
    result = query_capabilities("/dev/video0", runner=lambda device: None)

    assert result.available is False
    assert result.device == "/dev/video0"
    assert result.capabilities == []
    assert "v4l2-ctl" in result.reason


def test_query_capabilities_uses_injected_runner() -> None:
    seen: list[str] = []

    def runner(device: str) -> str:
        seen.append(device)
        return V4L2_TEXT

    result = query_capabilities("/dev/video2", runner=runner)

    assert seen == ["/dev/video2"]
    assert result.available is True
    assert result.capabilities[0].fps_list == [144, 120, 60]
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/test_capture_caps.py -q
```

Expected: import failure for `novasight.capture`.

- [ ] **Step 3: Implement capture state and caps parser**

Create `novasight/capture/state.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


CapturePreference = Literal[
    "auto_high_fps",
    "auto_low_latency",
    "auto_balanced",
    "manual",
]


@dataclass(frozen=True)
class CaptureCapability:
    pixel_format: str
    width: int
    height: int
    fps_list: list[int]


@dataclass(frozen=True)
class CaptureCapabilities:
    available: bool
    device: str
    capabilities: list[CaptureCapability] = field(default_factory=list)
    reason: str = ""


@dataclass(frozen=True)
class CaptureProfile:
    device: str
    pixel_format: str
    width: int
    height: int
    fps: int
    preference: CapturePreference
    selection_reason: str


@dataclass
class CaptureRuntimeState:
    available: bool = False
    device: str = "/dev/video0"
    profile: CaptureProfile | None = None
    backend: str | None = None
    fps_capture: float = 0.0
    frame_period_ms: float = 0.0
    capture_wait_ms: float = 0.0
    frames_dropped: int = 0
    recoveries: int = 0
    last_error: str | None = None
```

Create `novasight/capture/caps.py`:

```python
from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Callable

from .state import CaptureCapabilities, CaptureCapability


_FORMAT_RE = re.compile(r"^\s*\[\d+\]:\s*'([A-Za-z0-9]+)'")
_SIZE_RE = re.compile(r"^\s*Size:\s*Discrete\s+(\d+)x(\d+)")
_FPS_RE = re.compile(r"\(([\d.]+)\s*fps\)")


def run_v4l2_ctl(device: str) -> str | None:
    if shutil.which("v4l2-ctl") is None:
        return None
    try:
        result = subprocess.run(
            ["v4l2-ctl", "-d", device, "--list-formats-ext"],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def parse_v4l2_formats(text: str) -> list[CaptureCapability]:
    capabilities: list[CaptureCapability] = []
    current_format: str | None = None
    current_width: int | None = None
    current_height: int | None = None
    current_fps: list[int] = []

    def flush() -> None:
        nonlocal current_width, current_height, current_fps
        if current_format and current_width and current_height and current_fps:
            capabilities.append(
                CaptureCapability(
                    pixel_format=current_format.upper(),
                    width=current_width,
                    height=current_height,
                    fps_list=sorted(set(current_fps), reverse=True),
                )
            )
        current_width = None
        current_height = None
        current_fps = []

    for line in text.splitlines():
        if match := _FORMAT_RE.match(line):
            flush()
            current_format = match.group(1).upper()
            continue
        if match := _SIZE_RE.match(line):
            flush()
            current_width = int(match.group(1))
            current_height = int(match.group(2))
            continue
        if match := _FPS_RE.search(line):
            current_fps.append(int(round(float(match.group(1)))))

    flush()
    return capabilities


def query_capabilities(
    device: str = "/dev/video0",
    runner: Callable[[str], str | None] = run_v4l2_ctl,
) -> CaptureCapabilities:
    text = runner(device)
    if text is None:
        return CaptureCapabilities(
            available=False,
            device=device,
            reason="v4l2-ctl unavailable, device missing, or command failed",
        )
    capabilities = parse_v4l2_formats(text)
    return CaptureCapabilities(
        available=bool(capabilities),
        device=device,
        capabilities=capabilities,
        reason="" if capabilities else "no discrete V4L2 formats found",
    )
```

Create `novasight/capture/__init__.py`:

```python
from .caps import parse_v4l2_formats, query_capabilities
from .state import (
    CaptureCapabilities,
    CaptureCapability,
    CapturePreference,
    CaptureProfile,
    CaptureRuntimeState,
)

__all__ = [
    "CaptureCapabilities",
    "CaptureCapability",
    "CapturePreference",
    "CaptureProfile",
    "CaptureRuntimeState",
    "parse_v4l2_formats",
    "query_capabilities",
]
```

- [ ] **Step 4: Run tests**

Run:

```bash
pytest tests/test_capture_caps.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add novasight/capture tests/test_capture_caps.py
git commit -m "feat: add v4l2 capture capability parser"
```

---

### Task 2: Capture Profile Selection

**Files:**
- Create: `novasight/capture/profile.py`
- Modify: `novasight/capture/__init__.py`
- Test: `tests/test_capture_profile.py`

- [ ] **Step 1: Write failing profile tests**

Create `tests/test_capture_profile.py`:

```python
import pytest

from novasight.capture import CaptureCapability
from novasight.capture.profile import select_capture_profile


CAPS = [
    CaptureCapability("MJPG", 1920, 1080, [144, 120, 60]),
    CaptureCapability("NV12", 1920, 1080, [60]),
    CaptureCapability("YUYV", 1280, 720, [60]),
    CaptureCapability("MJPG", 3840, 2160, [30]),
]


def test_auto_high_fps_prefers_1080p_144_mjpg() -> None:
    profile = select_capture_profile("/dev/video0", CAPS, "auto_high_fps")

    assert profile.pixel_format == "MJPG"
    assert profile.width == 1920
    assert profile.height == 1080
    assert profile.fps == 144
    assert "highest fps" in profile.selection_reason


def test_auto_low_latency_prefers_nv12_when_fps_is_usable() -> None:
    profile = select_capture_profile("/dev/video0", CAPS, "auto_low_latency")

    assert profile.pixel_format == "NV12"
    assert profile.width == 1920
    assert profile.height == 1080
    assert profile.fps == 60


def test_auto_balanced_prefers_1080p_before_4k30() -> None:
    profile = select_capture_profile("/dev/video0", CAPS, "auto_balanced")

    assert profile.width == 1920
    assert profile.height == 1080
    assert profile.fps >= 60


def test_manual_requires_exact_supported_profile() -> None:
    profile = select_capture_profile(
        "/dev/video1",
        CAPS,
        "manual",
        pixel_format="NV12",
        width=1920,
        height=1080,
        fps=60,
    )

    assert profile.device == "/dev/video1"
    assert profile.pixel_format == "NV12"


def test_manual_rejects_unsupported_profile() -> None:
    with pytest.raises(ValueError, match="unsupported capture profile"):
        select_capture_profile(
            "/dev/video0",
            CAPS,
            "manual",
            pixel_format="NV12",
            width=1920,
            height=1080,
            fps=144,
        )
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/test_capture_profile.py -q
```

Expected: import failure for `novasight.capture.profile`.

- [ ] **Step 3: Implement selector**

Create `novasight/capture/profile.py`:

```python
from __future__ import annotations

from .state import CaptureCapability, CapturePreference, CaptureProfile


def _expanded(caps: list[CaptureCapability]) -> list[tuple[str, int, int, int]]:
    return [
        (cap.pixel_format.upper(), cap.width, cap.height, fps)
        for cap in caps
        for fps in cap.fps_list
    ]


def _format_rank_high_fps(pixel_format: str, fps: int) -> int:
    if pixel_format == "MJPG" and fps >= 120:
        return 0
    if pixel_format == "NV12" and fps >= 60:
        return 1
    if pixel_format == "YUYV":
        return 2
    if pixel_format == "MJPG":
        return 3
    return 4


def _format_rank_low_latency(pixel_format: str) -> int:
    return {"NV12": 0, "YUYV": 1, "MJPG": 2}.get(pixel_format, 3)


def select_capture_profile(
    device: str,
    capabilities: list[CaptureCapability],
    preference: CapturePreference = "auto_high_fps",
    *,
    pixel_format: str | None = None,
    width: int | None = None,
    height: int | None = None,
    fps: int | None = None,
) -> CaptureProfile:
    choices = _expanded(capabilities)
    if not choices:
        raise ValueError(f"no capture capabilities available for {device}")

    if preference == "manual":
        required = (str(pixel_format).upper(), width, height, fps)
        for choice in choices:
            if choice == required:
                return CaptureProfile(
                    device=device,
                    pixel_format=choice[0],
                    width=choice[1],
                    height=choice[2],
                    fps=choice[3],
                    preference=preference,
                    selection_reason="manual profile matched device capabilities",
                )
        raise ValueError(f"unsupported capture profile for {device}: {required}")

    if preference == "auto_low_latency":
        choice = sorted(
            choices,
            key=lambda item: (
                _format_rank_low_latency(item[0]),
                -item[3],
                -(item[1] * item[2]),
            ),
        )[0]
        reason = "auto_low_latency selected raw-friendly profile"
    elif preference == "auto_balanced":
        choice = sorted(
            choices,
            key=lambda item: (
                abs((item[1] * item[2]) - (1920 * 1080)),
                -item[3],
                _format_rank_high_fps(item[0], item[3]),
            ),
        )[0]
        reason = "auto_balanced selected profile closest to 1080p with high fps"
    else:
        choice = sorted(
            choices,
            key=lambda item: (
                -item[3],
                _format_rank_high_fps(item[0], item[3]),
                -(item[1] * item[2]),
            ),
        )[0]
        reason = "auto_high_fps selected highest fps profile"

    return CaptureProfile(
        device=device,
        pixel_format=choice[0],
        width=choice[1],
        height=choice[2],
        fps=choice[3],
        preference=preference,
        selection_reason=reason,
    )
```

Modify `novasight/capture/__init__.py` to export `select_capture_profile`.

- [ ] **Step 4: Run tests**

Run:

```bash
pytest tests/test_capture_caps.py tests/test_capture_profile.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add novasight/capture tests/test_capture_profile.py
git commit -m "feat: select capture profiles from v4l2 capabilities"
```

---

### Task 3: Capture Pipeline Candidates and Source Opening

**Files:**
- Create: `novasight/capture/pipeline.py`
- Create: `novasight/capture/source.py`
- Modify: `novasight/capture/__init__.py`
- Test: `tests/test_capture_pipeline.py`

- [ ] **Step 1: Write failing pipeline tests**

Create `tests/test_capture_pipeline.py`:

```python
from novasight.capture import CaptureProfile
from novasight.capture.pipeline import build_pipeline_candidates, select_open_source


def _profile(fmt: str = "MJPG") -> CaptureProfile:
    return CaptureProfile(
        device="/dev/video0",
        pixel_format=fmt,
        width=1920,
        height=1080,
        fps=144 if fmt == "MJPG" else 60,
        preference="auto_high_fps",
        selection_reason="test",
    )


def test_mjpg_candidates_start_with_nvmm_decoder() -> None:
    candidates = build_pipeline_candidates(_profile("MJPG"))

    assert candidates[0].label == "gst:nvmm-mjpg-iomode2"
    assert "v4l2src device=/dev/video0" in candidates[0].pipeline
    assert "image/jpeg,width=1920,height=1080,framerate=144/1" in candidates[0].pipeline
    assert "nvv4l2decoder mjpeg=1" in candidates[0].pipeline
    assert candidates[-1].label == "opencv:v4l2"


def test_nv12_candidates_start_with_nvmm_nv12() -> None:
    candidates = build_pipeline_candidates(_profile("NV12"))

    assert candidates[0].label == "gst:nvmm-nv12-iomode2"
    assert "video/x-raw,format=NV12,width=1920,height=1080,framerate=60/1" in candidates[0].pipeline


def test_select_open_source_returns_first_candidate_that_reads() -> None:
    attempts: list[str] = []

    def opener(candidate):
        attempts.append(candidate.label)
        return candidate.label == "gst:cpu-jpegdec-mjpg"

    selected = select_open_source(_profile("MJPG"), opener=opener)

    assert selected.label == "gst:cpu-jpegdec-mjpg"
    assert attempts[:4] == [
        "gst:nvmm-mjpg-iomode2",
        "gst:nvmm-mjpg-iomode4",
        "gst:nvmm-mjpg-ioauto",
        "gst:cpu-jpegdec-mjpg",
    ]
    assert selected.failures == attempts[:3]


def test_select_open_source_raises_when_all_candidates_fail() -> None:
    try:
        select_open_source(_profile("MJPG"), opener=lambda candidate: False)
    except RuntimeError as exc:
        assert "no capture backend opened" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/test_capture_pipeline.py -q
```

Expected: import failure for `novasight.capture.pipeline`.

- [ ] **Step 3: Implement pipeline candidate builder**

Create `novasight/capture/pipeline.py` with:

```python
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .state import CaptureProfile


@dataclass(frozen=True)
class CaptureCandidate:
    label: str
    pipeline: str


@dataclass(frozen=True)
class SelectedCaptureBackend:
    label: str
    pipeline: str
    failures: list[str] = field(default_factory=list)


def build_pipeline_candidates(profile: CaptureProfile) -> list[CaptureCandidate]:
    device = profile.device
    width = profile.width
    height = profile.height
    fps = profile.fps
    fmt = profile.pixel_format.upper()
    sink = "appsink drop=true max-buffers=1 sync=false"

    mjpg_caps = f"image/jpeg,width={width},height={height},framerate={fps}/1"
    nv12_caps = f"video/x-raw,format=NV12,width={width},height={height},framerate={fps}/1"
    yuyv_caps = f"video/x-raw,format=YUY2,width={width},height={height},framerate={fps}/1"

    def gst(label: str, body: str) -> CaptureCandidate:
        return CaptureCandidate(label=label, pipeline=body)

    mjpg = [
        gst("gst:nvmm-mjpg-iomode2", f"v4l2src device={device} io-mode=2 ! {mjpg_caps} ! jpegparse ! nvv4l2decoder mjpeg=1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! {sink}"),
        gst("gst:nvmm-mjpg-iomode4", f"v4l2src device={device} io-mode=4 ! {mjpg_caps} ! jpegparse ! nvv4l2decoder mjpeg=1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! {sink}"),
        gst("gst:nvmm-mjpg-ioauto", f"v4l2src device={device} ! {mjpg_caps} ! jpegparse ! nvv4l2decoder mjpeg=1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! {sink}"),
        gst("gst:cpu-jpegdec-mjpg", f"v4l2src device={device} ! {mjpg_caps} ! jpegdec ! videoconvert ! video/x-raw,format=BGR ! {sink}"),
    ]
    nv12 = [
        gst("gst:nvmm-nv12-iomode2", f"v4l2src device={device} io-mode=2 ! {nv12_caps} ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! {sink}"),
        gst("gst:nvmm-nv12-iomode4", f"v4l2src device={device} io-mode=4 ! {nv12_caps} ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! {sink}"),
        gst("gst:nvmm-nv12-ioauto", f"v4l2src device={device} ! {nv12_caps} ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! {sink}"),
        gst("gst:cpu-nv12-videoconvert", f"v4l2src device={device} ! {nv12_caps} ! videoconvert ! video/x-raw,format=BGR ! {sink}"),
    ]
    yuyv = [
        gst("gst:nvmm-yuyv-iomode2", f"v4l2src device={device} io-mode=2 ! {yuyv_caps} ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! {sink}"),
        gst("gst:cpu-yuyv-videoconvert", f"v4l2src device={device} ! {yuyv_caps} ! videoconvert ! video/x-raw,format=BGR ! {sink}"),
    ]

    ordered = mjpg + nv12 + yuyv
    if fmt == "NV12":
        ordered = nv12 + yuyv + mjpg
    elif fmt == "YUYV":
        ordered = yuyv + nv12 + mjpg
    return ordered + [CaptureCandidate(label="opencv:v4l2", pipeline="")]


def select_open_source(
    profile: CaptureProfile,
    opener: Callable[[CaptureCandidate], bool],
) -> SelectedCaptureBackend:
    failures: list[str] = []
    for candidate in build_pipeline_candidates(profile):
        if opener(candidate):
            return SelectedCaptureBackend(
                label=candidate.label,
                pipeline=candidate.pipeline,
                failures=failures,
            )
        failures.append(candidate.label)
    raise RuntimeError(f"no capture backend opened for {profile.device}: {', '.join(failures)}")
```

Create `novasight/capture/source.py` with protocol and lazy `cv2` import:

```python
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

from .pipeline import CaptureCandidate
from .state import CaptureProfile


@dataclass(frozen=True)
class CapturedFrame:
    frame_id: int
    width: int
    height: int
    pixel_format: str
    ts_ns: int
    capture_wait_ms: float
    image: Any


class FrameSource(Protocol):
    backend_label: str

    def read(self) -> CapturedFrame | None: ...
    def close(self) -> None: ...


class OpenCvFrameSource:
    def __init__(self, profile: CaptureProfile, candidate: CaptureCandidate) -> None:
        import cv2

        self.profile = profile
        self.backend_label = candidate.label
        self._cv2 = cv2
        if candidate.label == "opencv:v4l2":
            self._cap = cv2.VideoCapture(profile.device, cv2.CAP_V4L2)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, profile.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, profile.height)
            self._cap.set(cv2.CAP_PROP_FPS, profile.fps)
        else:
            self._cap = cv2.VideoCapture(candidate.pipeline, cv2.CAP_GSTREAMER)
        self._frame_id = 0

    def opened_and_readable(self) -> bool:
        if not self._cap.isOpened():
            self.close()
            return False
        ok, image = self._cap.read()
        if not ok or image is None:
            self.close()
            return False
        return True

    def read(self) -> CapturedFrame | None:
        t0 = time.monotonic_ns()
        ok, image = self._cap.read()
        t1 = time.monotonic_ns()
        if not ok or image is None:
            return None
        self._frame_id += 1
        return CapturedFrame(
            frame_id=self._frame_id,
            width=int(image.shape[1]),
            height=int(image.shape[0]),
            pixel_format="BGR",
            ts_ns=t1,
            capture_wait_ms=(t1 - t0) / 1e6,
            image=image,
        )

    def close(self) -> None:
        self._cap.release()
```

Modify `novasight/capture/__init__.py` to export candidate/source types.

- [ ] **Step 4: Run tests**

Run:

```bash
pytest tests/test_capture_pipeline.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add novasight/capture tests/test_capture_pipeline.py
git commit -m "feat: build jetson capture pipeline candidates"
```

---

### Task 4: Capture Service, Runtime Config, and CLI Diagnostics

**Files:**
- Create: `novasight/capture/service.py`
- Modify: `novasight/config/runtime.py`
- Modify: `config/novasight.example.yaml`
- Modify: `novasight/main.py`
- Test: `tests/test_capture_service.py`

- [ ] **Step 1: Write failing service and CLI tests**

Create `tests/test_capture_service.py`:

```python
from novasight.capture import CaptureCapability
from novasight.capture.service import CaptureService
from novasight.config import RuntimeConfig


class FakeSource:
    backend_label = "gst:test"

    def __init__(self) -> None:
        self.count = 0

    def read(self):
        from novasight.capture.source import CapturedFrame

        self.count += 1
        return CapturedFrame(
            frame_id=self.count,
            width=1920,
            height=1080,
            pixel_format="BGR",
            ts_ns=1_000_000_000 + self.count * 7_000_000,
            capture_wait_ms=6.5,
            image=None,
        )

    def close(self) -> None:
        pass


def test_capture_service_selects_profile_from_config() -> None:
    cfg = RuntimeConfig()
    service = CaptureService(
        config=cfg.capture,
        capability_runner=lambda device: "[0]: 'MJPG' (Motion-JPEG)\n    Size: Discrete 1920x1080\n        Interval: Discrete 0.007s (144.000 fps)\n",
        source_factory=lambda profile: FakeSource(),
    )

    state = service.configure("/dev/video0")

    assert state.available is True
    assert state.profile is not None
    assert state.profile.pixel_format == "MJPG"
    assert state.profile.fps == 144
    assert state.backend == "gst:test"


def test_capture_service_smoke_updates_timing() -> None:
    cfg = RuntimeConfig()
    service = CaptureService(
        config=cfg.capture,
        capability_runner=lambda device: "[0]: 'NV12' (Y/CbCr)\n    Size: Discrete 1920x1080\n        Interval: Discrete 0.017s (60.000 fps)\n",
        source_factory=lambda profile: FakeSource(),
    )
    service.configure("/dev/video0")

    state = service.capture_frames(max_frames=3)

    assert state.fps_capture > 0
    assert state.capture_wait_ms == 6.5
    assert state.frame_period_ms > 0
    assert state.last_error is None
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/test_capture_service.py -q
```

Expected: import failure for `novasight.capture.service` or missing `RuntimeConfig.capture`.

- [ ] **Step 3: Add runtime capture config**

Modify `novasight/config/runtime.py`:

```python
@dataclass
class CaptureConfig:
    device: str = "/dev/video0"
    preference: str = "auto_high_fps"
    pixel_format: str = ""
    width: int = 0
    height: int = 0
    fps: int = 0


@dataclass
class ControlConfig:
    max_abs_dx: int = 120
    max_abs_dy: int = 120
    min_confidence: int = 0
    output_mode: str = "silent"
```

Add fields to `RuntimeConfig`:

```python
capture: CaptureConfig = field(default_factory=CaptureConfig)
control: ControlConfig = field(default_factory=ControlConfig)
```

Use `int` fields only in this task because the existing config validator
supports `int` and `str`. Interpret `min_confidence` as percentage later if
needed; Task 8 can broaden validation to float in a controlled change.

- [ ] **Step 4: Implement capture service**

Create `novasight/capture/service.py`:

```python
from __future__ import annotations

import time
from collections.abc import Callable

from novasight.config.runtime import CaptureConfig

from .caps import query_capabilities, run_v4l2_ctl
from .profile import select_capture_profile
from .source import FrameSource
from .state import CaptureProfile, CaptureRuntimeState


class CaptureService:
    def __init__(
        self,
        config: CaptureConfig,
        *,
        capability_runner: Callable[[str], str | None] | None = None,
        source_factory: Callable[[CaptureProfile], FrameSource] | None = None,
    ) -> None:
        self.config = config
        self.capability_runner = capability_runner
        self.source_factory = source_factory
        self.state = CaptureRuntimeState(device=config.device)
        self.source: FrameSource | None = None

    def configure(self, device: str | None = None) -> CaptureRuntimeState:
        selected_device = device or self.config.device
        caps = query_capabilities(
            selected_device,
            runner=self.capability_runner or run_v4l2_ctl,
        )
        if not caps.available:
            self.state = CaptureRuntimeState(
                available=False,
                device=selected_device,
                last_error=caps.reason,
            )
            return self.state
        profile = select_capture_profile(
            selected_device,
            caps.capabilities,
            self.config.preference,  # type: ignore[arg-type]
            pixel_format=self.config.pixel_format or None,
            width=self.config.width or None,
            height=self.config.height or None,
            fps=self.config.fps or None,
        )
        source = self.source_factory(profile) if self.source_factory else None
        self.source = source
        self.state = CaptureRuntimeState(
            available=source is not None,
            device=selected_device,
            profile=profile,
            backend=source.backend_label if source else None,
            last_error=None if source else "capture source factory unavailable",
        )
        return self.state

    def capture_frames(
        self,
        *,
        seconds: float | None = None,
        max_frames: int | None = None,
    ) -> CaptureRuntimeState:
        if self.source is None:
            raise RuntimeError("capture source is not configured")
        start = time.monotonic()
        previous_ts: int | None = None
        count = 0
        while True:
            if max_frames is not None and count >= max_frames:
                break
            if seconds is not None and time.monotonic() - start >= seconds:
                break
            frame = self.source.read()
            if frame is None:
                self.state.frames_dropped += 1
                continue
            count += 1
            self.state.capture_wait_ms = frame.capture_wait_ms
            if previous_ts is not None:
                self.state.frame_period_ms = (frame.ts_ns - previous_ts) / 1e6
            previous_ts = frame.ts_ns
        elapsed = max(time.monotonic() - start, 0.000001)
        self.state.fps_capture = count / elapsed
        return self.state
```

- [ ] **Step 5: Add CLI subcommands**

Modify `novasight/main.py` so `parse_args()` supports:

```python
subparsers = parser.add_subparsers(dest="command")
doctor = subparsers.add_parser("doctor")
doctor_sub = doctor.add_subparsers(dest="doctor_command")
doctor_camera = doctor_sub.add_parser("camera")
doctor_camera.add_argument("--device", default="/dev/video0")

capture_smoke = subparsers.add_parser("capture-smoke")
capture_smoke.add_argument("--device", default="/dev/video0")
capture_smoke.add_argument("--seconds", type=float, default=5.0)
```

In `main()`, before starting Uvicorn:

```python
if args.command == "doctor" and args.doctor_command == "camera":
    from novasight.capture import query_capabilities
    from novasight.capture.profile import select_capture_profile

    caps = query_capabilities(args.device)
    print(f"device: {caps.device}")
    print(f"available: {caps.available}")
    for cap in caps.capabilities:
        print(f"{cap.pixel_format} {cap.width}x{cap.height} fps={cap.fps_list}")
    if caps.available:
        profile = select_capture_profile(args.device, caps.capabilities)
        print(f"selected: {profile.pixel_format} {profile.width}x{profile.height}@{profile.fps}")
    elif caps.reason:
        print(f"reason: {caps.reason}")
    return 0 if caps.available else 2
```

Implement `capture-smoke` similarly after Task 3 source factory exists.

- [ ] **Step 6: Run tests**

Run:

```bash
pytest tests/test_config_runtime.py tests/test_capture_service.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add novasight/config/runtime.py config/novasight.example.yaml novasight/main.py novasight/capture tests/test_capture_service.py
git commit -m "feat: add capture service and diagnostics"
```

---

### Task 5: Capture API and Runtime State

**Files:**
- Create: `novasight/api/routes_capture.py`
- Modify: `novasight/api/app.py`
- Modify: `novasight/runtime/state.py`
- Modify: `novasight/runtime/service.py`
- Modify: `novasight/api/routes_health.py`
- Test: `tests/test_capture_api.py`
- Test: `tests/test_api.py`

- [ ] **Step 1: Write failing API tests**

Create `tests/test_capture_api.py`:

```python
from fastapi.testclient import TestClient

from novasight.api import create_app


def test_capture_state_is_in_runtime_state(tmp_path) -> None:
    client = TestClient(create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml"))

    response = client.get("/api/runtime/state")

    assert response.status_code == 200
    assert "capture" in response.json()
    assert response.json()["capture"]["device"] == "/dev/video0"


def test_capture_capabilities_endpoint_uses_service(tmp_path) -> None:
    client = TestClient(create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml"))

    response = client.get("/api/capture/capabilities?device=/dev/video0")

    assert response.status_code == 200
    body = response.json()
    assert body["device"] == "/dev/video0"
    assert "available" in body
    assert "capabilities" in body


def test_capture_select_rejects_unavailable_device(tmp_path) -> None:
    client = TestClient(create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml"))

    response = client.post("/api/capture/select", json={"device": "/dev/missing"})

    assert response.status_code in {200, 400}
    assert "device" in response.json()
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/test_capture_api.py -q
```

Expected: 404 for `/api/capture/capabilities`.

- [ ] **Step 3: Add capture router**

Create `novasight/api/routes_capture.py`:

```python
from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Request
from pydantic import BaseModel

from novasight.capture import query_capabilities


router = APIRouter(prefix="/api/capture", tags=["capture"])


class CaptureSelectRequest(BaseModel):
    device: str = "/dev/video0"


@router.get("/capabilities")
def capabilities(device: str = "/dev/video0") -> dict:
    return asdict(query_capabilities(device))


@router.get("/state")
def state(request: Request) -> dict:
    return asdict(request.app.state.capture.state)


@router.post("/select")
def select(request: Request, payload: CaptureSelectRequest) -> dict:
    state = request.app.state.capture.configure(payload.device)
    return asdict(state)
```

Modify `novasight/api/app.py`:

```python
from novasight.capture.service import CaptureService
from .routes_capture import router as capture_router
...
capture = CaptureService(config.capture)
app.state.capture = capture
app.include_router(capture_router)
```

- [ ] **Step 4: Expand runtime state**

Modify `novasight/runtime/state.py`:

```python
@dataclass(frozen=True)
class RuntimeState:
    running: bool
    source: str
    active_model: dict | None
    executor: dict
    capture: dict
    inference: dict
```

Modify `RuntimeService.state()`:

```python
capture_state = getattr(self, "capture", None)
inference_state = getattr(self, "inference", None)
...
capture=asdict(capture_state.state) if capture_state is not None else {},
inference=inference_state.status() if inference_state is not None else {"available": False},
```

Pass capture/inference services through constructor with defaults so old tests can be updated cleanly.

- [ ] **Step 5: Run tests**

Run:

```bash
pytest tests/test_api.py tests/test_capture_api.py -q
```

Expected: all tests pass after updating existing assertions to tolerate `capture` and `inference` keys.

- [ ] **Step 6: Commit**

```bash
git add novasight/api novasight/runtime tests/test_api.py tests/test_capture_api.py
git commit -m "feat: expose capture state api"
```

---

### Task 6: Inference Boundary and TensorRT Unavailable State

**Files:**
- Create: `novasight/inference/__init__.py`
- Create: `novasight/inference/contracts.py`
- Create: `novasight/inference/unavailable.py`
- Create: `novasight/inference/tensorrt.py`
- Create: `novasight/inference/runtime.py`
- Modify: `novasight/api/app.py`
- Modify: `novasight/runtime/service.py`
- Test: `tests/test_inference_runtime.py`

- [ ] **Step 1: Write failing inference tests**

Create `tests/test_inference_runtime.py`:

```python
from pathlib import Path

from novasight.capture.source import CapturedFrame
from novasight.inference import InferenceDetection, InferenceResult, UnavailableInferenceEngine


def test_unavailable_engine_reports_reason_and_returns_empty_result() -> None:
    engine = UnavailableInferenceEngine("TensorRT unavailable")

    assert engine.available() is False
    assert engine.status()["reason"] == "TensorRT unavailable"

    result = engine.infer(
        CapturedFrame(1, 640, 480, "BGR", 123, 1.0, image=None)
    )

    assert result.available is False
    assert result.detections == []
    assert result.reason == "TensorRT unavailable"


def test_inference_result_detection_shape() -> None:
    result = InferenceResult(
        available=True,
        detections=[
            InferenceDetection(cls=0, score=0.9, x=10, y=20, w=30, h=40),
        ],
        classes=["target"],
    )

    assert result.detections[0].x == 10
    assert result.classes == ["target"]
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/test_inference_runtime.py -q
```

Expected: import failure for `novasight.inference`.

- [ ] **Step 3: Implement inference contracts**

Create `novasight/inference/contracts.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from novasight.capture.source import CapturedFrame


@dataclass(frozen=True)
class InferenceDetection:
    cls: int
    score: float
    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True)
class InferenceResult:
    available: bool
    detections: list[InferenceDetection] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    reason: str = ""


class InferenceEngine(Protocol):
    engine_id: str

    def available(self) -> bool: ...
    def status(self) -> dict: ...
    def load(self, artifact_path: Path, classes: list[str], input_shape: str) -> None: ...
    def infer(self, frame: CapturedFrame) -> InferenceResult: ...
```

Create `novasight/inference/unavailable.py`:

```python
from __future__ import annotations

from pathlib import Path

from novasight.capture.source import CapturedFrame

from .contracts import InferenceResult


class UnavailableInferenceEngine:
    engine_id = "unavailable"

    def __init__(self, reason: str = "inference engine unavailable") -> None:
        self.reason = reason

    def available(self) -> bool:
        return False

    def status(self) -> dict:
        return {"selected": self.engine_id, "available": False, "reason": self.reason}

    def load(self, artifact_path: Path, classes: list[str], input_shape: str) -> None:
        raise RuntimeError(self.reason)

    def infer(self, frame: CapturedFrame) -> InferenceResult:
        return InferenceResult(available=False, reason=self.reason)
```

Create `novasight/inference/tensorrt.py` with lazy import:

```python
from __future__ import annotations

from pathlib import Path

from novasight.capture.source import CapturedFrame

from .contracts import InferenceResult


class TensorRtInferenceEngine:
    engine_id = "tensorrt"

    def __init__(self) -> None:
        self._reason = ""
        self._loaded = False

    def available(self) -> bool:
        try:
            import tensorrt  # noqa: F401
        except Exception as exc:
            self._reason = f"TensorRT unavailable: {exc}"
            return False
        return True

    def status(self) -> dict:
        return {
            "selected": self.engine_id,
            "available": self.available(),
            "loaded": self._loaded,
            "reason": self._reason,
        }

    def load(self, artifact_path: Path, classes: list[str], input_shape: str) -> None:
        if not artifact_path.suffix == ".engine":
            raise ValueError(f"TensorRT artifact must be .engine: {artifact_path}")
        if not self.available():
            raise RuntimeError(self._reason)
        self._loaded = True

    def infer(self, frame: CapturedFrame) -> InferenceResult:
        if not self._loaded:
            return InferenceResult(available=False, reason="TensorRT engine not loaded")
        return InferenceResult(available=True, detections=[], classes=[])
```

Create `novasight/inference/__init__.py` exports.

- [ ] **Step 4: Add runtime resolver skeleton**

Create `novasight/inference/runtime.py`:

```python
from __future__ import annotations

from .contracts import InferenceEngine
from .tensorrt import TensorRtInferenceEngine
from .unavailable import UnavailableInferenceEngine


class InferenceRuntime:
    def __init__(self, engine: InferenceEngine | None = None) -> None:
        self.engine = engine or TensorRtInferenceEngine()
        if not self.engine.available():
            self.engine = UnavailableInferenceEngine(self.engine.status().get("reason", "TensorRT unavailable"))

    def status(self) -> dict:
        return self.engine.status()
```

Attach `InferenceRuntime` in `create_app()` and `RuntimeService`.

- [ ] **Step 5: Run tests**

Run:

```bash
pytest tests/test_inference_runtime.py tests/test_api.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add novasight/inference novasight/api/app.py novasight/runtime tests/test_inference_runtime.py
git commit -m "feat: add inference runtime boundary"
```

---

### Task 7: Runtime Frame Flow From Capture to Plugins

**Files:**
- Modify: `novasight/runtime/service.py`
- Modify: `novasight/plugins/contracts.py`
- Test: `tests/test_inference_runtime.py`
- Test: `tests/test_plugin_executor_flow.py`

- [ ] **Step 1: Write failing captured-frame flow test**

Append to `tests/test_inference_runtime.py`:

```python
from novasight.inference import InferenceDetection, InferenceResult
from novasight.runtime import RuntimeService
from novasight.config import RuntimeConfig
from novasight.model_registry import ModelRegistry
from novasight.plugins import PluginRuntime
from novasight.executors import ExecutorRegistry
from novasight.executors.dry_run import DryRunExecutor


class FakeInferenceRuntime:
    def status(self) -> dict:
        return {"selected": "fake", "available": True}

    @property
    def engine(self):
        return self

    def infer(self, frame):
        return InferenceResult(
            available=True,
            detections=[InferenceDetection(0, 0.9, 10, 20, 30, 40)],
            classes=["target"],
        )


def test_runtime_process_captured_frame_converts_inference_to_context(tmp_path):
    service = RuntimeService(
        config=RuntimeConfig(),
        models=ModelRegistry(tmp_path / "db.sqlite", tmp_path / "models"),
        plugins=PluginRuntime.with_builtin_plugins(),
        executors=ExecutorRegistry([DryRunExecutor()]),
        inference=FakeInferenceRuntime(),
    )

    frame = CapturedFrame(7, 640, 480, "BGR", 123, 1.0, image=None)
    result = service.process_captured_frame(frame)

    assert result.plugin_batch.plugin_results
    assert result.plugin_batch.control_intents
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
pytest tests/test_inference_runtime.py::test_runtime_process_captured_frame_converts_inference_to_context -q
```

Expected: `RuntimeService` has no `process_captured_frame`.

- [ ] **Step 3: Implement captured-frame processing**

Modify `novasight/runtime/service.py`:

```python
from novasight.capture.source import CapturedFrame
from novasight.plugins import Detection
...
def process_captured_frame(self, frame: CapturedFrame) -> RuntimeFrameResult:
    if self.inference is None:
        context = FrameContext(frame.frame_id, frame.width, frame.height)
    else:
        inference_result = self.inference.engine.infer(frame)
        detections = [
            Detection(
                cls=item.cls,
                score=item.score,
                x=item.x,
                y=item.y,
                w=item.w,
                h=item.h,
            )
            for item in inference_result.detections
        ]
        context = FrameContext(
            frame_id=frame.frame_id,
            width=frame.width,
            height=frame.height,
            detections=detections,
            classes=inference_result.classes,
        )
    return self.process_frame(context)
```

- [ ] **Step 4: Run tests**

Run:

```bash
pytest tests/test_inference_runtime.py tests/test_plugin_executor_flow.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add novasight/runtime tests/test_inference_runtime.py tests/test_plugin_executor_flow.py
git commit -m "feat: route captured frames through inference and plugins"
```

---

### Task 8: Experimental Plugin Pair

**Files:**
- Modify: `novasight/plugins/builtin.py`
- Modify: `novasight/plugins/runtime.py`
- Test: `tests/test_experimental_plugins.py`

- [ ] **Step 1: Write failing plugin tests**

Create `tests/test_experimental_plugins.py`:

```python
from novasight.plugins import Detection, FrameContext, PluginRuntime


def test_experimental_target_reports_normalized_offset() -> None:
    runtime = PluginRuntime.with_builtin_plugins()
    context = FrameContext(
        frame_id=1,
        width=1000,
        height=500,
        detections=[Detection(cls=0, score=0.8, x=600, y=200, w=100, h=100)],
        classes=["target"],
    )

    result = runtime.process(context)
    target = next(item for item in result.plugin_results if item.plugin_id == "vision.experimental_target")

    assert target.payload["class_name"] == "target"
    assert target.payload["center"] == {"x": 650.0, "y": 250.0}
    assert target.payload["normalized_offset"] == {"x": 0.3, "y": 0.0}


def test_experimental_center_control_outputs_pixel_delta() -> None:
    runtime = PluginRuntime.with_builtin_plugins()
    context = FrameContext(
        frame_id=1,
        width=1000,
        height=500,
        detections=[Detection(cls=0, score=0.8, x=600, y=200, w=100, h=100)],
        classes=["target"],
    )

    intent = next(item for item in runtime.process(context).control_intents if item.plugin_id == "control.experimental_center")

    assert intent.dx == 150
    assert intent.dy == 0
    assert intent.confidence == 0.8
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/test_experimental_plugins.py -q
```

Expected: no plugin with id `vision.experimental_target`.

- [ ] **Step 3: Implement plugins**

Add to `novasight/plugins/builtin.py`:

```python
class ExperimentalTargetPlugin:
    plugin_id = "vision.experimental_target"
    kind = "vision"

    def process(self, context: FrameContext) -> PluginResult:
        target = max(context.detections, key=lambda item: item.score, default=None)
        if target is None:
            return PluginResult(self.plugin_id, self.kind, {"target": None})
        cx = target.cx
        cy = target.cy
        class_name = context.classes[target.cls] if 0 <= target.cls < len(context.classes) else str(target.cls)
        return PluginResult(
            plugin_id=self.plugin_id,
            kind=self.kind,
            payload={
                "class_name": class_name,
                "score": target.score,
                "center": {"x": cx, "y": cy},
                "normalized_offset": {
                    "x": round((cx - context.width / 2) / (context.width / 2), 6),
                    "y": round((context.height / 2 - cy) / (context.height / 2), 6),
                },
                "reason": "highest-score detection",
            },
        )


class ExperimentalCenterControlPlugin:
    plugin_id = "control.experimental_center"
    kind = "control"

    def process(self, context: FrameContext) -> ControlIntent | None:
        target = max(context.detections, key=lambda item: item.score, default=None)
        if target is None:
            return None
        return ControlIntent(
            dx=target.cx - context.width / 2,
            dy=context.height / 2 - target.cy,
            action="move",
            confidence=target.score,
            reason="experimental center target",
            plugin_id=self.plugin_id,
        )
```

Modify `PluginRuntime.with_builtin_plugins()` to include the two plugins.

- [ ] **Step 4: Run tests**

Run:

```bash
pytest tests/test_experimental_plugins.py tests/test_plugin_executor_flow.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add novasight/plugins tests/test_experimental_plugins.py
git commit -m "feat: add experimental target plugins"
```

---

### Task 9: Bounded Control Output Modes

**Files:**
- Create: `novasight/control/__init__.py`
- Create: `novasight/control/output.py`
- Modify: `novasight/config/runtime.py`
- Modify: `novasight/executors/contracts.py`
- Modify: `novasight/executors/dry_run.py`
- Modify: `novasight/executors/runtime.py`
- Test: `tests/test_control_output.py`
- Test: `tests/test_plugin_executor_flow.py`

- [ ] **Step 1: Write failing control output tests**

Create `tests/test_control_output.py`:

```python
from novasight.control import ControlOutputPolicy
from novasight.plugins import ControlIntent


def _intent(dx=500, dy=-300, confidence=0.9):
    return ControlIntent(dx, dy, "move", confidence, "test", "control.test")


def test_policy_clamps_dx_dy() -> None:
    output = ControlOutputPolicy(max_abs_dx=120, max_abs_dy=80).apply(_intent())

    assert output.dx == 120
    assert output.dy == -80
    assert output.clipped is True
    assert "clamped" in output.reason


def test_policy_drops_low_confidence() -> None:
    output = ControlOutputPolicy(min_confidence=0.5).apply(_intent(confidence=0.1))

    assert output.accepted is False
    assert output.dx == 0
    assert output.dy == 0
    assert "confidence" in output.reason


def test_policy_rejects_non_finite_values() -> None:
    output = ControlOutputPolicy().apply(_intent(dx=float("nan")))

    assert output.accepted is False
    assert "non-finite" in output.reason
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/test_control_output.py -q
```

Expected: import failure for `novasight.control`.

- [ ] **Step 3: Implement control output policy**

Create `novasight/control/output.py`:

```python
from __future__ import annotations

import math
from dataclasses import dataclass

from novasight.plugins import ControlIntent


@dataclass(frozen=True)
class ControlOutput:
    dx: int
    dy: int
    action: str | None
    confidence: float
    plugin_id: str
    accepted: bool
    clipped: bool
    reason: str


class ControlOutputPolicy:
    def __init__(
        self,
        *,
        max_abs_dx: int = 120,
        max_abs_dy: int = 120,
        min_confidence: float = 0.0,
    ) -> None:
        self.max_abs_dx = max_abs_dx
        self.max_abs_dy = max_abs_dy
        self.min_confidence = min_confidence

    def apply(self, intent: ControlIntent) -> ControlOutput:
        if not math.isfinite(intent.dx) or not math.isfinite(intent.dy):
            return ControlOutput(0, 0, intent.action, intent.confidence, intent.plugin_id, False, False, "non-finite control delta")
        if intent.confidence < self.min_confidence:
            return ControlOutput(0, 0, intent.action, intent.confidence, intent.plugin_id, False, False, "confidence below threshold")
        dx = max(-self.max_abs_dx, min(self.max_abs_dx, int(round(intent.dx))))
        dy = max(-self.max_abs_dy, min(self.max_abs_dy, int(round(intent.dy))))
        clipped = dx != int(round(intent.dx)) or dy != int(round(intent.dy))
        reason = "clamped to configured limits" if clipped else intent.reason
        return ControlOutput(dx, dy, intent.action, intent.confidence, intent.plugin_id, True, clipped, reason)
```

Create `novasight/control/__init__.py` exports.

- [ ] **Step 4: Route executor registry through policy**

Modify `ExecutorRegistry.__init__`:

```python
from novasight.control import ControlOutputPolicy
...
policy: ControlOutputPolicy | None = None,
...
self.policy = policy or ControlOutputPolicy()
```

Modify `execute()`:

```python
bounded = self.policy.apply(intent)
return self.executors[self.selected].execute(bounded)
```

Update `Executor` protocol and executor implementations to accept `ControlOutput`. Keep `ExecutionResult.intent` only if existing API tests require it, or rename to `output` and update tests in the same commit.

- [ ] **Step 5: Add silent/console modes**

Implement output modes as executors:

```python
class SilentExecutor:
    executor_id = "silent"
    def __init__(self) -> None:
        self.history: list[ControlOutput] = []
    def available(self) -> bool:
        return True
    def execute(self, output: ControlOutput) -> ExecutionResult:
        self.history.append(output)
        return ExecutionResult("silent", False, output, "swallowed")
```

`ConsoleExecutor` prints:

```text
control dx=12 dy=-4 action=move confidence=0.91 plugin=control.experimental_center accepted=True
```

Register `silent`, `console`, `dry_run`, and `kmnet`.

- [ ] **Step 6: Run tests**

Run:

```bash
pytest tests/test_control_output.py tests/test_plugin_executor_flow.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add novasight/control novasight/config/runtime.py novasight/executors tests/test_control_output.py tests/test_plugin_executor_flow.py
git commit -m "feat: bound control outputs before execution"
```

---

### Task 10: Web Capture Status and Documentation

**Files:**
- Modify: `web/src/api.ts`
- Modify: `web/src/App.tsx`
- Modify: `web/src/styles.css`
- Modify: `README.md`
- Test: `pnpm --dir web typecheck`
- Test: `pnpm --dir web build`
- Test: `pytest -q`

- [ ] **Step 1: Add web API types**

Modify `web/src/api.ts` with:

```ts
export type CaptureState = {
  available: boolean;
  device: string;
  profile: null | {
    pixel_format: string;
    width: number;
    height: number;
    fps: number;
    preference: string;
    selection_reason: string;
  };
  backend: string | null;
  fps_capture: number;
  frame_period_ms: number;
  capture_wait_ms: number;
  frames_dropped: number;
  recoveries: number;
  last_error: string | null;
};
```

Ensure the runtime state type includes:

```ts
capture: CaptureState;
inference: Record<string, unknown>;
```

- [ ] **Step 2: Show capture status in the existing console**

Modify `web/src/App.tsx` so Dashboard or Settings includes compact rows:

```tsx
<section className="panel">
  <h2>Capture</h2>
  <dl className="metric-list">
    <div><dt>Device</dt><dd>{state.capture.device}</dd></div>
    <div><dt>Backend</dt><dd>{state.capture.backend ?? "not selected"}</dd></div>
    <div>
      <dt>Profile</dt>
      <dd>
        {state.capture.profile
          ? `${state.capture.profile.pixel_format} ${state.capture.profile.width}x${state.capture.profile.height}@${state.capture.profile.fps}`
          : "not configured"}
      </dd>
    </div>
    <div><dt>FPS</dt><dd>{state.capture.fps_capture.toFixed(1)}</dd></div>
    <div><dt>Frame period</dt><dd>{state.capture.frame_period_ms.toFixed(2)} ms</dd></div>
    <div><dt>Capture wait</dt><dd>{state.capture.capture_wait_ms.toFixed(2)} ms</dd></div>
  </dl>
</section>
```

Keep styling compact; do not add large hero, marketing blocks, or decorative cards.

- [ ] **Step 3: Update README**

Add:

```markdown
## Jetson Camera Diagnostics

Inspect `/dev/video0` capabilities:

```bash
python3 -m novasight doctor camera --device /dev/video0
```

Run a short real capture smoke test:

```bash
python3 -m novasight capture-smoke --device /dev/video0 --seconds 5
```

The smoke test reports selected profile, backend label, observed FPS,
frame period, capture wait, dropped frames, recoveries, and recent errors.
```

- [ ] **Step 4: Run full verification**

Run:

```bash
pytest -q
pnpm --dir web typecheck
pnpm --dir web build
```

Expected:

```text
pytest: all tests pass
tsc: exits 0
vite build: exits 0
```

- [ ] **Step 5: Commit**

```bash
git add web README.md
git commit -m "docs: surface capture diagnostics"
```

---

## Final Verification

- [ ] **Step 1: Run Python tests**

```bash
pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Run web checks**

```bash
pnpm --dir web typecheck
pnpm --dir web build
```

Expected: both commands exit 0.

- [ ] **Step 3: Run backend smoke**

```bash
python3 -m novasight --host 127.0.0.1 --port 5174
```

In another shell:

```bash
curl -s http://127.0.0.1:5174/healthz
curl -s http://127.0.0.1:5174/api/runtime/state
curl -s http://127.0.0.1:5174/api/capture/state
```

Expected: health is `{"ok":true}`, runtime state contains `capture`, and capture state contains `device`.

- [ ] **Step 4: Jetson-only smoke**

On Jetson with a camera or capture card attached:

```bash
python3 -m novasight doctor camera --device /dev/video0
python3 -m novasight capture-smoke --device /dev/video0 --seconds 5
```

Expected: `doctor camera` prints supported formats and the selected profile;
`capture-smoke` prints backend label, observed FPS, frame period, and capture wait.
