from pathlib import Path


STUDIO_CONSOLE = (
    Path(__file__).resolve().parents[1]
    / "web"
    / "src"
    / "features"
    / "studio"
    / "StudioConsoleView.tsx"
)
DEVICES_VIEW = (
    Path(__file__).resolve().parents[1]
    / "web"
    / "src"
    / "features"
    / "devices"
    / "DevicesView.tsx"
)


def _roi_card_source() -> str:
    source = STUDIO_CONSOLE.read_text(encoding="utf-8")
    start = source.index('<SectionTitle title="ROI 裁剪" />')
    end = source.index('<NumberControl label="水平偏移"', start)
    return source[start:end]


def test_roi_size_control_commits_once_instead_of_writing_on_every_change() -> None:
    roi_card = _roi_card_source()

    assert 'onChange={(event) => void updateConfigField("roi", "size"' not in roi_card
    assert 'onCommit={(value) => updateConfigField("roi", "size", nearestRoiSize(value))}' in roi_card


def test_studio_mainline_mode_includes_tensorrt_runtime_backend() -> None:
    source = STUDIO_CONSOLE.read_text(encoding="utf-8")

    assert 'selectedRuntimeBackend === "nvmm_latest"' not in source
    assert 'new Set(["nvmm_latest", "tensorrt"])' in source


def test_devices_mainline_mode_includes_tensorrt_runtime_backend() -> None:
    source = DEVICES_VIEW.read_text(encoding="utf-8")

    assert 'selected === "nvmm_latest"' not in source
    assert 'inferenceSelected === "nvmm_latest"' not in source
    assert 'new Set(["nvmm_latest", "tensorrt"])' in source


def test_kmnet_panel_has_dedicated_control_test_page() -> None:
    source = STUDIO_CONSOLE.read_text(encoding="utf-8")
    section_start = source.index(
        'activePage === "params" || activePage === "control-test"'
    )
    control_page_start = source.index(
        '<Metric title="连接状态" value={kmnetConnected ? "已连接" : kmnetConnecting ? "连接中" : "未连接"}',
        section_start,
    )
    section_end = source.index("</section>", control_page_start)

    assert '{ id: "control-test", index: "05", label: "控制测试", icon: "kmbox" }' in source
    assert '<SectionTitle title="kmNet 控制面板" />' not in source[section_start:control_page_start]
    assert '<SectionTitle title="kmNet 控制面板" />' in source[control_page_start:section_end]


def test_kmnet_connection_button_uses_backend_runtime_state() -> None:
    source = STUDIO_CONSOLE.read_text(encoding="utf-8")

    assert "const kmnetConnected = kmnetStatus.connected === true;" in source
    assert "const kmnetConnecting = kmnetStatus.connecting === true;" in source
    assert '{kmnetConnected ? "断开 kmNet" : kmnetConnecting ? "取消连接 kmNet" : "连接 kmNet"}' in source
    assert 'enabled={kmnetAutoConnect}' in source
    assert 'updateConfigField("hardware", "auto_connect", enabled)' in source


def test_studio_diagnostics_separate_capture_inference_and_control_layers() -> None:
    source = STUDIO_CONSOLE.read_text(encoding="utf-8")
    capture_start = source.index('activePage === "capture" ? "console-page active"')
    inference_start = source.index('activePage === "infer" ? "console-page active"')
    control_start = source.index('activePage === "control" ? "console-page active"')
    params_start = source.index('activePage === "params" || activePage === "control-test"')
    capture_page = source[capture_start:inference_start]
    inference_page = source[inference_start:control_start]
    control_page = source[control_start:params_start]

    assert 'data-layer="capture"' in capture_page
    assert '<SectionTitle title="最新帧状态" />' in capture_page
    assert '<SectionTitle title="采集性能" />' in capture_page
    assert '模型输入尺寸' not in capture_page
    assert '理论 counts' not in capture_page

    assert 'data-layer="inference"' in inference_page
    assert '<SectionTitle title="推理调度" />' in inference_page
    assert '<SectionTitle title="TensorRT 执行" />' in inference_page
    assert 'GPU 等待' not in inference_page
    assert '目标框中心' not in inference_page
    assert '瞄准点' not in inference_page
    assert '理论 counts' not in inference_page

    assert 'data-layer="control"' in control_page
    for title in (
        "目标选择",
        "瞄准点与预测",
        "误差与角度",
        "控制器输出",
        "Scheduler 与设备发送",
    ):
        assert f'<SectionTitle title="{title}" />' in control_page


def test_studio_telemetry_uses_explicit_missing_and_capability_states() -> None:
    source = STUDIO_CONSOLE.read_text(encoding="utf-8")

    assert 'const NO_SAMPLE = "—";' in source
    assert 'const NOT_INSTRUMENTED = "未接入";' in source
    assert 'const UNAVAILABLE = "不可用";' in source


def test_studio_tracker_controls_match_active_hungarian_mainline() -> None:
    source = STUDIO_CONSOLE.read_text(encoding="utf-8")

    assert '<span>关联算法</span><b>Hungarian</b>' in source
    assert '<span>输出状态</span><b>仅 ACTIVE</b>' in source
    assert 'updateConfigField("control", "tracker_max_match_distance", value)' in source
    assert 'updateConfigField("control", "tracker_max_missed_frames", Math.round(value))' in source
    assert "experimental_angle_hungarian_enabled" not in source
    assert 'label="Track 确认帧数"' not in source
    assert 'label="Track 匹配距离 px"' not in source


def test_studio_exposes_only_mutually_exclusive_control_modes() -> None:
    studio = STUDIO_CONSOLE.read_text(encoding="utf-8")
    devices = DEVICES_VIEW.read_text(encoding="utf-8")

    assert '<option value="universal_saturated">通用适配</option>' in studio
    assert '<option value="calibrated_angular">精确标定</option>' in studio
    assert 'updateConfigField("control", "mode", event.target.value)' in studio
    assert 'updateControlGroupField("calibrated_angular", "kp_x", value)' in studio
    assert 'updateControlGroupField("universal_saturated", "response_scale_x_px", value)' in studio
    assert 'updateControlGroupField("shared", "max_count_slew_x", value)' in studio
    assert "experimental_angle" not in studio
    assert "experimental_angle" not in devices
    assert 'SettingsSection = "capture" | "inference";' in devices
