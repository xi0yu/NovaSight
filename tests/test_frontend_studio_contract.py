from pathlib import Path


STUDIO_CONSOLE = (
    Path(__file__).resolve().parents[1]
    / "web"
    / "src"
    / "features"
    / "studio"
    / "StudioConsoleView.tsx"
)
APP = STUDIO_CONSOLE.parents[2] / "App.tsx"
STUDIO_NAVIGATION = (
    Path(__file__).resolve().parents[1]
    / "web"
    / "src"
    / "features"
    / "studio"
    / "StudioNavigation.tsx"
)
STUDIO_CONTROLS = (
    Path(__file__).resolve().parents[1]
    / "web"
    / "src"
    / "features"
    / "studio"
    / "StudioControls.tsx"
)
STUDIO_SETTINGS = (
    Path(__file__).resolve().parents[1]
    / "web"
    / "src"
    / "features"
    / "studio"
    / "studio-settings.css"
)
MODEL_SELECTION = (
    STUDIO_CONSOLE.parents[1] / "models" / "ModelSelectionPanel.tsx"
)
LICENSE_VIEW = (
    STUDIO_CONSOLE.parents[1] / "license" / "LicenseView.tsx"
)
ADVANCED_SETTINGS_DIALOG = (
    STUDIO_CONSOLE.parent / "AdvancedSettingsDialog.tsx"
)
STUDIO_PRESENTATION = (
    STUDIO_CONSOLE.parent / "StudioPresentation.tsx"
)
RUNTIME_STATUS = (
    Path(__file__).resolve().parents[1] / "web" / "src" / "features" / "shared" / "runtimeStatus.ts"
)


def _roi_card_source() -> str:
    source = STUDIO_CONSOLE.read_text(encoding="utf-8")
    start = source.index('<SectionTitle title="ROI 裁剪" />')
    end = source.index('<div className="console-kv compact-kv">', start)
    return source[start:end]


def test_roi_size_control_commits_once_instead_of_writing_on_every_change() -> None:
    roi_card = _roi_card_source()

    assert 'onChange={(event) => void updateConfigField("roi", "size"' not in roi_card
    assert (
        'onCommit={(value) => updateConfigField("roi", "size", nearestRoiSize(value))}' in roi_card
    )
    assert "ROI 模式" not in roi_card
    assert "水平偏移" not in roi_card
    assert "垂直偏移" not in roi_card


def test_studio_exposes_only_15_or_30_fps_for_inference_preview() -> None:
    source = STUDIO_CONSOLE.read_text(encoding="utf-8")

    assert 'aria-label="推理画面预览帧率"' in source
    assert "{[15, 30].map((fps)" in source
    assert "目标帧率" not in source


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


def test_mainline_launch_dialog_renders_every_step_with_explicit_state_icons() -> None:
    source = STUDIO_CONSOLE.read_text(encoding="utf-8")

    assert 'className="launch-stage-list"' in source
    assert "launchStages.map((stage, index)" in source
    assert 'type LaunchStepState = "pending" | "running" | "success" | "failed"' in source
    assert "<LaunchStepIndicator state={stepState} />" in source
    assert 'className="launch-step-arc"' in source
    assert 'status === "failed" && index === activeIndex' in source
    assert 'title: "激活鼠标算法"' in source
    assert "输出设备不影响本步骤" in source
    assert "确认设备执行器" not in source


def test_model_file_lists_show_human_readable_size() -> None:
    panel = MODEL_SELECTION.read_text(encoding="utf-8")
    catalog_tree = (
        STUDIO_CONSOLE.parents[1] / "models" / "ModelCatalogTree.tsx"
    ).read_text(encoding="utf-8")
    styles = (STUDIO_CONSOLE.parents[2] / "styles.css").read_text(encoding="utf-8")

    assert "formatModelSize(selectedModel?.size_bytes ?? selectedArtifact?.size_bytes)" in panel
    assert '<span className="model-catalog-size">{formatModelSize(node.size_bytes)}</span>' in catalog_tree
    assert "grid-template-columns: minmax(0, 1fr) auto;" in styles
    assert "font-variant-numeric: tabular-nums;" in styles


def test_studio_model_catalog_can_refresh_same_version_artifacts() -> None:
    source = STUDIO_CONSOLE.read_text(encoding="utf-8")

    assert "modelCatalogRefreshKey" in source
    assert "refreshModelCatalog" in source
    assert "selectedModelVersionId" in source


def test_studio_defers_non_capture_io_and_uses_launch_specific_status_timeout() -> None:
    source = STUDIO_CONSOLE.read_text(encoding="utf-8")
    api = (STUDIO_CONSOLE.parents[2] / "api.ts").read_text(encoding="utf-8")

    catalog_effect = source[source.index("setModelCatalogLoading(true)") - 180:]
    catalog_effect = catalog_effect[:catalog_effect.index("setModelCatalogLoading(false)")]
    assert 'if (activePage !== "infer")' in catalog_effect
    assert "void refreshCapabilities();" not in source
    assert "LAUNCH_STATUS_REQUEST_TIMEOUT_MS" in source
    assert "getRuntimeState(undefined, LAUNCH_STATUS_REQUEST_TIMEOUT_MS)" in source
    assert "timeoutMs = STATUS_REQUEST_TIMEOUT_MS" in api


def test_studio_refreshes_changed_models_without_exposing_force_revalidation() -> None:
    source = STUDIO_CONSOLE.read_text(encoding="utf-8")
    panel = MODEL_SELECTION.read_text(encoding="utf-8")
    refresh_start = source.index("const refreshModelCatalog")
    handlers_end = source.index("const launchStages", refresh_start)

    assert "getModelCatalog(false)" in source[refresh_start:handlers_end]
    assert "getModelCatalog(true)" not in source
    assert "rescanModelCatalog" not in source
    assert "scanModelDirectory" not in source
    assert '"刷新模型"' in panel
    assert '"强制重新校验"' not in source


def test_studio_registers_unmanaged_catalog_engine_by_reference() -> None:
    source = STUDIO_CONSOLE.read_text(encoding="utf-8")
    selection_start = source.index("const selectModelFromCatalog")
    selection_end = source.index("const switchModel", selection_start)
    selection = source[selection_start:selection_end]

    assert "registerCatalogModel(model.relative_path)" in selection
    assert "尚未登记完成，请重新扫描" not in selection
    assert "引用原始 Engine" in selection


def test_studio_auto_configures_and_switches_pending_engine() -> None:
    source = STUDIO_CONSOLE.read_text(encoding="utf-8")
    panel = MODEL_SELECTION.read_text(encoding="utf-8")

    assert 'item.status === "ready" || item.status === "pending" || item.status === "failed"' in source
    assert 'selectedArtifact?.status === "pending" || selectedArtifact?.status === "failed"' in panel
    assert "publishModel(" in source
    assert "parserPreset" in source
    assert "inspectModelArtifact" not in source
    assert "configureModelProfile" not in source
    assert "probeModelArtifact" not in source
    assert '"自动配置并加载模型"' in panel
    assert "自动生成唯一 DeepStream manifest" in panel
    assert "yoloCandidateCount" not in source
    assert "preferLatestModelVersionRef.current" in source
    assert "模型产物不可切换" in panel


def test_studio_uses_truthful_runtime_metrics_and_explicit_auto_save_copy() -> None:
    source = STUDIO_CONSOLE.read_text(encoding="utf-8")
    app = APP.read_text(encoding="utf-8")
    license_view = LICENSE_VIEW.read_text(encoding="utf-8")
    advanced_dialog = ADVANCED_SETTINGS_DIALOG.read_text(encoding="utf-8")
    presentation = STUDIO_PRESENTATION.read_text(encoding="utf-8")

    assert '<Metric title="P95 延迟" value="待机"' not in source
    assert '<Metric title="运行时长"' not in source
    assert '<KvCard title="系统状态"' not in source
    assert "<Bar " not in source
    assert "latencyStages.map((stage)" in source
    assert 'width={30}' not in source
    assert '"暂无样本"' in presentation

    assert "serviceUnavailable" in license_view
    assert "无法读取 NovaSight 后端" in license_view
    assert "这里不是卡密错误" in license_view
    assert "正在连接 NovaSight 后端" in license_view
    assert "确认结果前不会显示卡密输入" in license_view
    assert "useState(true)" in app

    assert "正在自动保存并同步运行配置" in advanced_dialog
    assert "已自动保存" in advanced_dialog
    assert '>关闭</button>' in advanced_dialog
    assert "已自动保存 · 修改后立即生效" in source
    assert "已自动保存 · 当前配置" in source
    assert "pendingConfigWriteCount" in source
    assert "dialogSavingRef" in source
    assert "保存失败 · ${dialogSaveError}" in source
    assert 'inert: dialogSaving ? "" : undefined' in source


def test_model_selection_exposes_only_builtin_parser_presets() -> None:
    panel = (
        STUDIO_CONSOLE.parents[1] / "models" / "ModelSelectionPanel.tsx"
    ).read_text(encoding="utf-8")

    assert 'value="auto"' in panel
    assert 'value="yolov5"' in panel
    assert 'value="yolov8"' in panel
    assert 'value="yolo11"' in panel
    assert 'value="novasight_generic"' in panel
    assert "NovaSight 通用解析器（内置）" in panel
    assert "外部 .so" not in panel


def test_kmnet_panel_has_dedicated_control_test_page() -> None:
    source = STUDIO_CONSOLE.read_text(encoding="utf-8")
    navigation = STUDIO_NAVIGATION.read_text(encoding="utf-8")
    section_start = source.index('activePage === "params" || activePage === "control-test"')
    control_page_start = source.index('<SectionTitle title="kmNet 控制面板" />', section_start)
    section_end = source.index("</section>", control_page_start)

    assert 'label: "诊断工具"' in navigation
    assert '{ id: "control-test", label: "控制测试", detail: "硬件输出实验", icon: "kmbox" }' in navigation
    assert '<SectionTitle title="kmNet 控制面板" />' in source[control_page_start:section_end]
    assert "const kmnetConnected = kmnetStatus.connected === true;" in source
    assert "const kmnetConnecting = kmnetStatus.connecting === true;" in source
    assert "kmnetStatus.connection_state" in source
    assert "kmnetStatus.retryable" in source
    assert '? "重新连接"' in source
    assert 'aria-pressed={kmnetConnected}' in source
    assert "主链可继续运行，输出暂不可用" in source
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
    assert '<SectionTitle title="nvinfer 阶段" />' in inference_page
    assert "预处理 + TensorRT + parser" in inference_page
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
        '<SectionTitle title={dualPhaseActive ? "Latest Replace 与设备发送" '
        ': "Scheduler 与设备发送"} />'
    ) in control_page


def test_studio_telemetry_uses_explicit_missing_and_capability_states() -> None:
    source = STUDIO_CONSOLE.read_text(encoding="utf-8")

    assert 'const NO_SAMPLE = "—";' in source
    assert 'const UNAVAILABLE = "不可用";' in source
    assert 'const NOT_INSTRUMENTED = "未接入";' not in source
    assert "inflight 估计" not in source
    assert "发送频率" not in source


def test_studio_tracker_controls_match_active_hungarian_mainline() -> None:
    source = STUDIO_CONSOLE.read_text(encoding="utf-8")

    assert "<span>关联算法</span><b>Hungarian</b>" in source
    assert "<span>输出状态</span><b>仅 ACTIVE</b>" in source
    assert 'updateConfigField("control", "tracker_max_match_distance", value)' in source
    assert 'updateConfigField("control", "tracker_max_missed_frames", Math.round(value))' in source
    assert "experimental_angle_hungarian_enabled" not in source
    assert 'label="Track 确认帧数"' not in source
    assert 'label="Track 匹配距离 px"' not in source


def test_studio_class_editor_exposes_profile_scoped_roles_and_three_role_aim_range() -> None:
    source = STUDIO_CONSOLE.read_text(encoding="utf-8")
    controls = STUDIO_CONTROLS.read_text(encoding="utf-8")
    settings = STUDIO_SETTINGS.read_text(encoding="utf-8")

    assert 'aria-label="模型类别与瞄点设置"' in source
    assert "头部、身体或其他角色" in source
    assert "实际值仍相对于各自 bbox" in source
    assert "未知类别（cls ${classId}）" in source
    assert 'updateConfigField("inference", "detection_class_profiles"' in source
    assert 'updateControlGroupField("aim", "class_roles"' in source
    assert 'updateControlGroupField("aim", "role_y_ratios"' in source
    assert "<AimTargetRange" in source
    assert "recordArray(vision.detection_items)" in source
    assert 'updateConfigField("inference", "detection_class_priority", current.join(","))' in source
    assert "setClassPriorityPosition" in source
    assert "参与目标选择的类别" in source
    assert "toggleDetectionClass" in source
    assert "新建空白" in source
    assert "复制当前" in source
    assert "全部选择" in source
    assert "全部取消" in source
    assert '"detection_class_filter", "none"' in source
    assert "允许当前检测类别" in source
    assert "effective_class_filter" in source
    assert "rejected_class_ids" in source
    assert "仅选此类" not in source
    assert "ClassAimRatioControl" not in controls
    assert ".class-config-workspace" in settings
    assert "overflow-y: auto" in settings
    assert ".class-config-dialog-footer" in settings
    assert "z-index: 3" in settings
    assert 'updateDualPhasePath(["aim", "y_ratio"]' not in source


def test_studio_exposes_only_mutually_exclusive_control_modes() -> None:
    studio = STUDIO_CONSOLE.read_text(encoding="utf-8")

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
    assert 'label="前瞻帧数"' in studio
    assert 'label="速度平滑帧数"' in studio
    assert 'title="Y 轴后坐力前馈 · 所有控制算法"' in studio
    assert 'updateDualPhasePath(["prediction", "coefficient"]' not in studio
    assert 'updateDualPhasePath(["prediction", "actuation_delay_ms"]' not in studio
    assert 'updateControlGroupField("calibrated_angular", "kp_x", value)' in studio
    assert 'updateControlGroupField("universal_saturated", "response_scale_x_px", value)' in studio
    assert 'updateControlGroupField("shared", "max_count_slew_x", value)' in studio
    assert "experimental_angle" not in studio
