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
    Path(__file__).resolve().parents[1] / "web" / "src" / "features" / "devices" / "DevicesView.tsx"
)
DASHBOARD_VIEW = (
    Path(__file__).resolve().parents[1]
    / "web"
    / "src"
    / "features"
    / "dashboard"
    / "DashboardView.tsx"
)
RUNTIME_STATUS = (
    Path(__file__).resolve().parents[1] / "web" / "src" / "features" / "shared" / "runtimeStatus.ts"
)


def _roi_card_source() -> str:
    source = STUDIO_CONSOLE.read_text(encoding="utf-8")
    start = source.index('<SectionTitle title="ROI 裁剪" />')
    end = source.index('<NumberControl label="水平偏移"', start)
    return source[start:end]


def test_roi_size_control_commits_once_instead_of_writing_on_every_change() -> None:
    roi_card = _roi_card_source()

    assert 'onChange={(event) => void updateConfigField("roi", "size"' not in roi_card
    assert (
        'onCommit={(value) => updateConfigField("roi", "size", nearestRoiSize(value))}' in roi_card
    )


def test_studio_exposes_only_deepstream_runtime_backend() -> None:
    source = STUDIO_CONSOLE.read_text(encoding="utf-8")

    assert 'new Set(["deepstream_nvinfer"])' in source
    assert 'label: "CPU latest"' not in source
    assert 'label: "NVMM latest"' not in source


def test_runtime_status_does_not_treat_informational_reason_as_failure() -> None:
    source = RUNTIME_STATUS.read_text(encoding="utf-8")

    failure_start = source.index("const failureMessage =")
    failure_end = source.index("const publishedBatches", failure_start)
    failure_logic = source[failure_start:failure_end]
    assert (
        '(terminalError ? readString(inference.detail) || readString(inference.reason) : "")'
        in failure_logic
    )
    assert "readString(inference.reason) ||" not in failure_logic
    assert "consumedBatches > 0 || controlObservations > 0" in source


def test_studio_deepstream_preview_uses_backend_hardware_jpeg_status() -> None:
    source = STUDIO_CONSOLE.read_text(encoding="utf-8")

    assert "const deepstreamPreviewStreamReady =" in source
    assert "runtimeInference.preview_enabled === true" in source
    assert "const previewImageAvailable = deepstreamNvinferSelected" in source
    assert '{showImage ? <img alt="实时画面 / ROI"' in source


def test_studio_preview_holds_one_transient_miss_and_keeps_box_nodes_stable() -> None:
    source = STUDIO_CONSOLE.read_text(encoding="utf-8")

    assert "PREVIEW_OVERLAY_HOLD_MS = 100" in source
    assert "useStablePreviewOverlay" in source
    assert "key={`box-${item.index}-${item.className}`}" in source
    assert "key={`box-${item.index}-${item.x}-${item.y}`}" not in source


def test_studio_preview_uses_roi_coordinates_for_control_center_and_aim_line() -> None:
    source = STUDIO_CONSOLE.read_text(encoding="utf-8")

    assert "predicted_aim_x_roi_px" in source
    assert "predicted_aim_y_roi_px" in source
    assert "sourceWidth / 2 - roiOffsetX" in source
    assert "sourceHeight / 2 - roiOffsetY" in source
    assert "left: percent(centerX, previewWidth)" in source
    assert "top: percent(centerY, previewHeight)" in source
    styles = (STUDIO_CONSOLE.parents[2] / "styles.css").read_text(encoding="utf-8")
    assert "aspect-ratio: var(--preview-aspect, 1 / 1);" in styles
    assert "object-fit: contain;" in styles
    target_lines_start = styles.rindex(".console-target-lines {")
    target_lines_end = styles.index("}", target_lines_start)
    target_lines_rule = styles[target_lines_start:target_lines_end]
    assert "width: 100%;" in target_lines_rule
    assert "height: 100%;" in target_lines_rule


def test_model_file_lists_show_size_in_megabytes() -> None:
    studio = STUDIO_CONSOLE.read_text(encoding="utf-8")
    models = (STUDIO_CONSOLE.parents[1] / "models" / "ModelsView.tsx").read_text(encoding="utf-8")

    assert "formatModelSizeMb(item.size_bytes)" in studio
    assert "formatModelSizeMb(artifact.size_bytes)" in models


def test_dashboard_deepstream_preview_uses_hardware_jpeg_branch() -> None:
    source = DASHBOARD_VIEW.read_text(encoding="utf-8")

    assert "const deepstreamPreviewStreamReady =" in source
    assert "runtimeInference.preview_enabled === true" in source
    assert "{previewImageAvailable ? (" in source
    assert "NVMM ROI · 硬件 JPEG" in source
    assert "Tensor Overlay" not in source
    assert 'readNumberRecord(targetMouseObservation, "predicted_aim_x_roi_px")' in source
    assert 'readNumberRecord(targetMouseObservation, "predicted_aim_y_roi_px")' in source


def test_studio_model_catalog_can_refresh_same_version_artifacts() -> None:
    source = STUDIO_CONSOLE.read_text(encoding="utf-8")

    assert "modelCatalogRefreshKey" in source
    assert "refreshModelCatalog" in source
    assert "[modelCatalogRefreshKey, selectedModelVersionId]" in source


def test_studio_refreshes_changed_models_and_can_force_revalidation() -> None:
    source = STUDIO_CONSOLE.read_text(encoding="utf-8")
    refresh_start = source.index("const refreshModelCatalog")
    rescan_start = source.index("const rescanModelCatalog", refresh_start)
    handlers_end = source.index("const launchStages", rescan_start)

    assert "scanModelDirectory(false)" in source[refresh_start:rescan_start]
    assert "scanModelDirectory(true)" in source[rescan_start:handlers_end]
    assert '"刷新模型"' in source
    assert '"强制重新校验"' in source


def test_studio_can_validate_and_switch_pending_engine() -> None:
    source = STUDIO_CONSOLE.read_text(encoding="utf-8")

    assert 'item.status === "ready" || item.status === "pending" || item.status === "failed"' in source
    assert 'selectedSwitchArtifact?.status === "pending"' in source
    assert "inspectModelArtifact" in source
    assert "configureModelProfile" in source
    assert "probeModelArtifact" in source
    assert '"检查、诊断并切换模型"' in source
    assert "无法从 Engine 自动确定的模型语义" in source
    assert "yoloCandidateCount" not in source
    assert "隔离诊断通过后才允许正式启用" in source
    assert "preferLatestModelVersionRef.current" in source
    assert "模型产物不可切换" in source


def test_devices_exposes_only_deepstream_runtime_backend() -> None:
    source = DEVICES_VIEW.read_text(encoding="utf-8")

    assert 'new Set(["deepstream_nvinfer"])' in source
    assert 'disabled={deepstreamNvinferSelected && id === "image"}' in source
    assert "DeepStream nvinfer 不支持" in source


def test_kmnet_panel_has_dedicated_control_test_page() -> None:
    source = STUDIO_CONSOLE.read_text(encoding="utf-8")
    section_start = source.index('activePage === "params" || activePage === "control-test"')
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
    assert (
        '{kmnetConnected ? "断开 kmNet" : kmnetConnecting ? "取消连接 kmNet" : "连接 kmNet"}'
        in source
    )
    assert "enabled={kmnetAutoConnect}" in source
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
    assert "模型输入尺寸" not in capture_page
    assert "理论 counts" not in capture_page

    assert 'data-layer="inference"' in inference_page
    assert '<SectionTitle title="推理调度" />' in inference_page
    assert '<SectionTitle title="TensorRT 执行" />' in inference_page
    assert "GPU 等待" not in inference_page
    assert "目标框中心" not in inference_page
    assert "瞄准点" not in inference_page
    assert "理论 counts" not in inference_page

    assert 'data-layer="control"' in control_page
    for title in (
        "目标选择",
        "瞄准点与预测",
        "误差与角度",
        "控制器输出",
    ):
        assert f'<SectionTitle title="{title}" />' in control_page
    assert (
        '<SectionTitle title={dualPhaseActive ? "MouseCommandExecutor 与设备发送" '
        ': "Scheduler 与设备发送"} />'
    ) in control_page


def test_studio_telemetry_uses_explicit_missing_and_capability_states() -> None:
    source = STUDIO_CONSOLE.read_text(encoding="utf-8")

    assert 'const NO_SAMPLE = "—";' in source
    assert 'const NOT_INSTRUMENTED = "未接入";' in source
    assert 'const UNAVAILABLE = "不可用";' in source


def test_studio_tracker_controls_match_active_hungarian_mainline() -> None:
    source = STUDIO_CONSOLE.read_text(encoding="utf-8")

    assert "<span>关联算法</span><b>Hungarian</b>" in source
    assert "<span>输出状态</span><b>仅 ACTIVE</b>" in source
    assert 'updateConfigField("control", "tracker_max_match_distance", value)' in source
    assert 'updateConfigField("control", "tracker_max_missed_frames", Math.round(value))' in source
    assert "experimental_angle_hungarian_enabled" not in source
    assert 'label="Track 确认帧数"' not in source
    assert 'label="Track 匹配距离 px"' not in source


def test_studio_exposes_only_mutually_exclusive_control_modes() -> None:
    studio = STUDIO_CONSOLE.read_text(encoding="utf-8")
    devices = DEVICES_VIEW.read_text(encoding="utf-8")

    assert 'id: "universal_saturated"' in studio
    assert 'label: "通用控制"' in studio
    assert 'id: "calibrated_angular"' in studio
    assert 'label: "精确角度控制"' in studio
    assert 'id: "dual_phase_atan_robust_predictive_v2"' in studio
    assert 'label: "稳健预测控制"' in studio
    assert 'updateConfigField("control", "active_algorithm", algorithm.id)' in studio
    assert "dual_phase_atan_predictive_v1" not in studio
    assert "ttbox_pid_atan" not in studio
    assert 'label="NEAR 阈值 px"' in studio
    assert 'label="共享 Atan 尺度 counts"' in studio
    assert 'updateControlGroupField("calibrated_angular", "kp_x", value)' in studio
    assert 'updateControlGroupField("universal_saturated", "response_scale_x_px", value)' in studio
    assert 'updateControlGroupField("shared", "max_count_slew_x", value)' in studio
    assert "experimental_angle" not in studio
    assert "experimental_angle" not in devices
    assert 'SettingsSection = "capture" | "inference";' in devices
