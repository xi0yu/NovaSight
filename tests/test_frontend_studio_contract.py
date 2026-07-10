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
        '<Metric title="连接状态" value={kmnetConnected ? "已连接" : "未连接"}',
        section_start,
    )
    section_end = source.index("</section>", control_page_start)

    assert '{ id: "control-test", index: "04", label: "控制测试", icon: "kmbox" }' in source
    assert '<SectionTitle title="kmNet 控制面板" />' not in source[section_start:control_page_start]
    assert '<SectionTitle title="kmNet 控制面板" />' in source[control_page_start:section_end]


def test_kmnet_connection_button_uses_backend_runtime_state() -> None:
    source = STUDIO_CONSOLE.read_text(encoding="utf-8")

    assert "const kmnetConnected = kmnetStatus.connected === true;" in source
    assert '{kmnetConnected ? "断开 kmNet" : "连接 kmNet"}' in source
    assert 'enabled={kmnetAutoConnect}' in source
    assert 'updateConfigField("hardware", "auto_connect", enabled)' in source
