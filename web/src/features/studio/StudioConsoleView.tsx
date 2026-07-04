import { ChangeEvent, Fragment, useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  CaptureCapabilitiesResponse,
  CaptureCapability,
  CaptureSelectPayload,
  HealthResponse,
  ModelArtifact,
  ModelProject,
  ModelVersion,
  RuntimeConfig,
  RuntimeState,
  getCaptureCapabilities,
  getModelArtifacts,
  getModelVersions,
  publishModel,
  selectCaptureProfile,
  stopCapture,
  streamUrl,
  updateRuntimeConfig
} from "../../api";
import { getErrorMessage } from "../shared/format";

type ConsolePage = "capture" | "infer" | "params" | "stats" | "latency";

type StudioConsoleViewProps = {
  health: HealthResponse | null;
  runtime: RuntimeState | null;
  projects: ModelProject[];
  errors: Partial<Record<string, string>>;
  lastUpdated: Date | null;
  onRefresh: () => Promise<void>;
};

type CapabilityChoice = {
  pixel_format: string;
  width: number;
  height: number;
  fps: number;
};

const navItems: { id: ConsolePage; index: string; label: string }[] = [
  { id: "capture", index: "01", label: "采集" },
  { id: "infer", index: "02", label: "模型推理" },
  { id: "params", index: "03", label: "参数设置" },
  { id: "stats", index: "04", label: "统计" },
  { id: "latency", index: "05", label: "采集延迟" }
];

const ROI_SIZE_CHOICES = [256, 320, 480, 640];

function asRecord(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function nestedRecord(value: unknown, key: string): Record<string, unknown> {
  return asRecord(asRecord(value)[key]);
}

function recordList(value: unknown): Record<string, string[]> {
  const raw = asRecord(value);
  return Object.fromEntries(
    Object.entries(raw)
      .filter(([, items]) => Array.isArray(items))
      .map(([key, items]) => [key, (items as unknown[]).map((item) => String(item))])
  );
}

function readNumber(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function readString(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

function formatNumber(value: unknown, digits = 1): string {
  const number = readNumber(value, Number.NaN);
  return Number.isFinite(number) ? number.toFixed(digits) : "待机";
}

function formatDate(value: Date | null): string {
  return value ? value.toLocaleTimeString("zh-CN", { hour12: false }) : "--:--:--";
}

function groupCapabilities(caps: CaptureCapability[]): CapabilityChoice[] {
  return caps
    .flatMap((cap) =>
      cap.fps_list.map((fps) => ({
        pixel_format: cap.pixel_format.toUpperCase(),
        width: cap.width,
        height: cap.height,
        fps
      }))
    )
    .sort((left, right) => {
      const formatRank = (value: string) => ["MJPG", "NV12", "YUYV", "BGR3"].indexOf(value);
      const leftRank = formatRank(left.pixel_format);
      const rightRank = formatRank(right.pixel_format);
      return (
        (leftRank < 0 ? 99 : leftRank) - (rightRank < 0 ? 99 : rightRank) ||
        right.fps - left.fps ||
        right.width * right.height - left.width * left.height
      );
    });
}

function choiceId(choice: CapabilityChoice): string {
  return `${choice.pixel_format}:${choice.width}x${choice.height}@${choice.fps}`;
}

function choiceLabel(choice: CapabilityChoice): string {
  return `${choice.pixel_format} / ${choice.width}x${choice.height} / ${choice.fps} FPS`;
}

function nearestRoiSize(value: number): number {
  return ROI_SIZE_CHOICES.reduce((best, current) =>
    Math.abs(current - value) < Math.abs(best - value) ? current : best
  );
}

function cloneRuntimeConfig(runtime: RuntimeState | null): RuntimeConfig | null {
  if (!runtime?.config) {
    return null;
  }
  const next = structuredClone(runtime.config) as RuntimeConfig;
  delete next.version;
  return next;
}

export function StudioConsoleView({
  health,
  runtime,
  projects,
  errors,
  lastUpdated,
  onRefresh
}: StudioConsoleViewProps) {
  const [activePage, setActivePage] = useState<ConsolePage>("capture");
  const [device, setDevice] = useState(runtime?.capture?.device ?? "/dev/video0");
  const [caps, setCaps] = useState<CaptureCapabilitiesResponse | null>(null);
  const [selectedChoiceId, setSelectedChoiceId] = useState("");
  const [selectedModelProjectId, setSelectedModelProjectId] = useState<number | "">("");
  const [selectedModelVersionId, setSelectedModelVersionId] = useState<number | "">("");
  const [selectedModelArtifactId, setSelectedModelArtifactId] = useState<number | "">("");
  const [modelVersions, setModelVersions] = useState<ModelVersion[]>([]);
  const [modelArtifacts, setModelArtifacts] = useState<ModelArtifact[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [localError, setLocalError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const capture = runtime?.capture;
  const statistics = runtime?.statistics ?? capture?.statistics;
  const config = runtime?.config;
  const roiConfig = nestedRecord(config, "roi");
  const inferenceConfig = nestedRecord(config, "inference");
  const controlConfig = nestedRecord(config, "control");
  const vision = asRecord(runtime?.vision);
  const inferenceTrace = asRecord(vision.inference);
  const pipeline = asRecord(runtime?.pipeline);
  const selectedProfile = capture?.profile;
  const choices = useMemo(() => groupCapabilities(caps?.capabilities ?? []), [caps]);
  const selectedChoice =
    choices.find((choice) => choiceId(choice) === selectedChoiceId) ?? choices[0];
  const roiSize = readNumber(roiConfig.size, 640);
  const confidence = readNumber(inferenceConfig.confidence_threshold, 0.25);
  const nms = readNumber(inferenceConfig.nms_threshold, 0.45);
  const detectionProfiles = recordList(inferenceConfig.detection_class_profiles);
  const activeDetectionProfile = readString(inferenceConfig.detection_class_profile, "default");
  const activeDetectionClass = readString(inferenceConfig.detection_class_filter, "all");
  const detectionProfileNames = Object.keys(detectionProfiles);
  const detectionClasses = detectionProfiles[activeDetectionProfile] ?? detectionProfiles.default ?? [];
  const pidKpX = readNumber(controlConfig.pid_kp_x, 0.35);
  const pidKpY = readNumber(controlConfig.pid_kp_y, 0.24);
  const pidKi = readNumber(controlConfig.pid_ki, 0.1);
  const pidKd = readNumber(controlConfig.pid_kd, 0.1);
  const kpXMoveMax = readNumber(controlConfig.kp_x_move_max, 150);
  const kpYMoveMax = readNumber(controlConfig.kp_y_move_max, 30);
  const activeModelName = runtime?.active_model?.project?.name ?? "未发布模型";
  const artifact = runtime?.active_model?.artifact;
  const version = runtime?.active_model?.version;
  const switchableArtifacts = modelArtifacts.filter(
    (item) => item.status === "ready" && (item.kind === "onnx" || item.kind === "engine")
  );
  const selectedSwitchArtifact =
    switchableArtifacts.find((item) => item.id === selectedModelArtifactId) ??
    switchableArtifacts[0] ??
    null;
  const detections = readNumber(vision.detections, 0);
  const target = asRecord(vision.target);
  const lastError = localError ?? Object.values(errors)[0] ?? capture?.last_error;

  useEffect(() => {
    if (runtime?.capture?.device) {
      setDevice(runtime.capture.device);
    }
  }, [runtime?.capture?.device]);

  useEffect(() => {
    if (!selectedChoiceId && selectedProfile) {
      setSelectedChoiceId(
        `${selectedProfile.pixel_format.toUpperCase()}:${selectedProfile.width}x${selectedProfile.height}@${selectedProfile.fps}`
      );
    }
  }, [selectedChoiceId, selectedProfile]);

  useEffect(() => {
    if (projects.length === 0) {
      setSelectedModelProjectId("");
      return;
    }
    setSelectedModelProjectId((current) =>
      typeof current === "number" && projects.some((project) => project.id === current)
        ? current
        : runtime?.active_model?.project?.id ?? projects[0].id
    );
  }, [projects, runtime?.active_model?.project?.id]);

  useEffect(() => {
    if (selectedModelProjectId === "") {
      setModelVersions([]);
      setSelectedModelVersionId("");
      return;
    }
    let cancelled = false;
    getModelVersions(selectedModelProjectId)
      .then((items) => {
        if (cancelled) {
          return;
        }
        setModelVersions(items);
        setSelectedModelVersionId((current) =>
          typeof current === "number" && items.some((item) => item.id === current)
            ? current
            : runtime?.active_model?.version?.id ?? items[0]?.id ?? ""
        );
      })
      .catch((err) => {
        if (!cancelled) {
          setModelVersions([]);
          setSelectedModelVersionId("");
          setLocalError(`模型版本读取失败：${getErrorMessage(err)}`);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [runtime?.active_model?.version?.id, selectedModelProjectId]);

  useEffect(() => {
    if (selectedModelVersionId === "") {
      setModelArtifacts([]);
      return;
    }
    let cancelled = false;
    getModelArtifacts(selectedModelVersionId)
      .then((items) => {
        if (cancelled) {
          return;
        }
        setModelArtifacts(items);
        const runnable = items.filter(
          (item) => item.status === "ready" && (item.kind === "onnx" || item.kind === "engine")
        );
        setSelectedModelArtifactId((current) =>
          typeof current === "number" && runnable.some((item) => item.id === current)
            ? current
            : artifact?.id ?? runnable[0]?.id ?? ""
        );
      })
      .catch((err) => {
        if (!cancelled) {
          setModelArtifacts([]);
          setLocalError(`模型产物读取失败：${getErrorMessage(err)}`);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [selectedModelVersionId]);

  const refreshCapabilities = useCallback(async () => {
    setBusy("caps");
    setLocalError(null);
    try {
      const result = await getCaptureCapabilities(device);
      setCaps(result);
      if (!result.available) {
        setLocalError(result.reason || "设备不可用");
      }
      const first = groupCapabilities(result.capabilities)[0];
      if (first) {
        setSelectedChoiceId(choiceId(first));
      }
    } catch (err) {
      setLocalError(getErrorMessage(err));
    } finally {
      setBusy(null);
    }
  }, [device]);

  useEffect(() => {
    void refreshCapabilities();
  }, [refreshCapabilities]);

  const applyCapture = useCallback(async () => {
    const choice = selectedChoice;
    setBusy("capture");
    setLocalError(null);
    const payload: CaptureSelectPayload = choice
      ? {
          device,
          preference: "manual",
          pixel_format: choice.pixel_format,
          width: choice.width,
          height: choice.height,
          fps: choice.fps
        }
      : selectedProfile
        ? {
            device,
            preference: "manual",
            pixel_format: selectedProfile.pixel_format,
            width: selectedProfile.width,
            height: selectedProfile.height,
            fps: selectedProfile.fps
          }
        : {
            device,
            preference: "auto_high_fps"
          };
    try {
      await selectCaptureProfile(payload);
      await onRefresh();
    } catch (err) {
      setLocalError(`切换失败，已保留上一组可用配置：${getErrorMessage(err)}`);
      await onRefresh();
    } finally {
      setBusy(null);
    }
  }, [device, onRefresh, selectedChoice, selectedProfile]);

  const stopCurrentCapture = useCallback(async () => {
    setBusy("stop");
    setLocalError(null);
    try {
      await stopCapture();
      await onRefresh();
    } catch (err) {
      setLocalError(getErrorMessage(err));
    } finally {
      setBusy(null);
    }
  }, [onRefresh]);

  const updateConfigField = useCallback(
    async (section: string, key: string, value: number | string | boolean) => {
      const next = cloneRuntimeConfig(runtime);
      if (!next) {
        return;
      }
      setBusy(`${section}.${key}`);
      setLocalError(null);
      const sectionValue = {
        ...asRecord(next[section])
      };
      sectionValue[key] = value;
      next[section] = sectionValue as RuntimeConfig[string];
      try {
        await updateRuntimeConfig(next);
        await onRefresh();
      } catch (err) {
        setLocalError(`配置同步失败：${getErrorMessage(err)}`);
      } finally {
        setBusy(null);
      }
    },
    [onRefresh, runtime]
  );

  const exportConfig = () => {
    if (!runtime?.config) {
      return;
    }
    const blob = new Blob([JSON.stringify(runtime.config, null, 2)], {
      type: "application/json"
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "novasight-config.json";
    link.click();
    URL.revokeObjectURL(url);
  };

  const importConfig = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) {
      return;
    }
    setBusy("import");
    setLocalError(null);
    try {
      const payload = JSON.parse(await file.text()) as RuntimeConfig;
      await updateRuntimeConfig(payload);
      await onRefresh();
    } catch (err) {
      setLocalError(`导入失败：${getErrorMessage(err)}`);
    } finally {
      setBusy(null);
    }
  };

  const switchModel = async () => {
    if (selectedModelProjectId === "" || selectedSwitchArtifact === null) {
      setLocalError("请选择可推理的 ONNX 或 TensorRT engine 产物。");
      return;
    }
    setBusy("model.switch");
    setLocalError(null);
    try {
      await publishModel(selectedModelProjectId, selectedSwitchArtifact.id);
      await onRefresh();
    } catch (err) {
      setLocalError(`模型切换失败，当前运行模型已保留：${getErrorMessage(err)}`);
      await onRefresh();
    } finally {
      setBusy(null);
    }
  };

  return (
    <section className="console-app">
      <header className="console-top">
        <div className="console-brand">
          <div className="console-logo" />
          NovaSight Studio
        </div>
        <div className="console-toolbar">
          <div className="console-group">
            <div>项目：{projects[0]?.name ?? "默认项目"}⌄</div>
            <div><span className="console-dot green" />设备：Jetson Orin Nano⌄</div>
          </div>
          <div className="console-group">
            <div><span className="console-dot" />{health?.ok ? "在线" : "离线"}</div>
            <div className="console-pill">退出</div>
            <div className="console-live">实时推送已连接 · {formatDate(lastUpdated)}</div>
          </div>
        </div>
      </header>

      <aside className="console-sidebar">
        {navItems.map((item) => (
          <button
            className={activePage === item.id ? "console-nav active" : "console-nav"}
            key={item.id}
            onClick={() => setActivePage(item.id)}
            type="button"
          >
            <span>{item.index}</span>
            <b>{item.label}</b>
          </button>
        ))}
      </aside>

      <main className="console-main">
        <section className="console-process">
          <div className="console-process-state">
            <span className="console-dot" />
            进程状态：{capture?.available ? "采集中" : "未运行"}
          </div>
          <button className="console-button primary" disabled={busy === "capture"} onClick={applyCapture} type="button">
            ▶ Start
          </button>
          <button className="console-button danger" disabled={busy === "stop"} onClick={stopCurrentCapture} type="button">
            ▪ Stop
          </button>
          <button className="console-button" onClick={exportConfig} type="button">导出配置...</button>
          <button className="console-button" onClick={() => fileInputRef.current?.click()} type="button">导入配置...</button>
          <input ref={fileInputRef} className="visually-hidden" type="file" accept="application/json,.json" onChange={importConfig} />
        </section>

        {lastError ? <div className="console-error">{lastError}</div> : null}

        <section className={activePage === "capture" ? "console-page active" : "console-page"}>
          <div className="console-metrics">
            <Metric title="采集 FPS" value={formatNumber(statistics?.capture_fps ?? capture?.fps_capture, 1)} small="FPS" />
            <Metric title="分辨率" value={selectedProfile ? `${selectedProfile.width}x${selectedProfile.height}` : "待机"} small="input" />
            <Metric title="像素格式" value={selectedProfile?.pixel_format ?? "待机"} small="format" />
            <Metric title="丢帧" value={String(statistics?.dropped_counter ?? capture?.frames_dropped ?? 0)} small="drop" />
          </div>

          <div className="console-grid2">
            <div>
              <div className="console-card">
                <h2 className="console-title">采集设备</h2>
                <label>视频设备</label>
                <input value={device} onChange={(event) => setDevice(event.target.value)} />
                <label>采集格式</label>
                <select value={selectedChoice ? choiceId(selectedChoice) : ""} onChange={(event) => setSelectedChoiceId(event.target.value)}>
                  {choices.map((choice) => (
                    <option key={choiceId(choice)} value={choiceId(choice)}>{choiceLabel(choice)}</option>
                  ))}
                  {choices.length === 0 ? <option>请先检测设备能力</option> : null}
                </select>
                <label>缓冲策略</label>
                <select value="latest-frame" disabled>
                  <option value="latest-frame">最新帧优先 / 单槽覆盖</option>
                </select>
                <div className="console-row">
                  <label>目标帧率</label>
                  <input value={selectedChoice?.fps ?? selectedProfile?.fps ?? ""} readOnly />
                </div>
                <button className="console-button primary" disabled={busy === "caps"} onClick={refreshCapabilities} type="button">
                  {busy === "caps" ? "检测中..." : "检测设备能力"}
                </button>
              </div>

              <div className="console-card">
                <h2 className="console-title">ROI 裁剪</h2>
                <label>ROI 模式</label>
                <select value="center" disabled>
                  <option value="center">中心正方形</option>
                </select>
                <label>ROI 尺寸</label>
                <div className="console-row">
                  <input
                    type="range"
                    min="256"
                    max="640"
                    step="64"
                    value={roiSize}
                    onChange={(event) => void updateConfigField("roi", "size", nearestRoiSize(Number(event.target.value)))}
                  />
                  <select value={roiSize} onChange={(event) => void updateConfigField("roi", "size", Number(event.target.value))}>
                    {ROI_SIZE_CHOICES.map((size) => <option key={size} value={size}>{size}</option>)}
                  </select>
                </div>
              </div>
            </div>
            <PreviewCard runtime={runtime} title="实时画面" roiSize={roiSize} />
          </div>
        </section>

        <section className={activePage === "infer" ? "console-page active" : "console-page"}>
          <div className="console-tabs">
            <div className="console-tab active"><span className="console-dot dark" />配置1</div>
            <div className="console-tab"><span className="console-dot" />配置2</div>
            <div className="console-tab"><span className="console-dot" />配置3</div>
          </div>
          <div className="console-metrics">
            <Metric title="推理 FPS" value={formatNumber(statistics?.inference_fps, 1)} small="FPS" />
            <Metric title="推理延迟" value={formatNumber(pipeline.e2e_latency_ms ?? statistics?.e2e_latency, 1)} small="ms" />
            <Metric title="目标数量" value={String(detections)} small="objects" />
            <Metric title="引擎状态" value={readString(runtime?.inference?.loaded, "") ? "已加载" : runtime?.inference?.loaded === true ? "已加载" : "未加载"} small={readString(runtime?.inference?.selected, "engine")} />
          </div>
          <div className="console-grid2">
            <div className="console-card">
              <h2 className="console-title">模型设置</h2>
              <label>模型文件</label>
              <select
                value={selectedModelProjectId}
                onChange={(event) => setSelectedModelProjectId(Number(event.target.value))}
              >
                {projects.map((project) => (
                  <option key={project.id} value={project.id}>{project.name}</option>
                ))}
                {projects.length === 0 ? <option value="">未发现模型</option> : null}
              </select>
              <label>模型版本</label>
              <select
                value={selectedModelVersionId}
                onChange={(event) => setSelectedModelVersionId(Number(event.target.value))}
              >
                {modelVersions.map((item) => (
                  <option key={item.id} value={item.id}>{item.version} · {item.input_shape}</option>
                ))}
                {modelVersions.length === 0 ? <option value="">暂无版本</option> : null}
              </select>
              <label>推理产物</label>
              <select
                value={selectedSwitchArtifact?.id ?? ""}
                onChange={(event) => setSelectedModelArtifactId(Number(event.target.value))}
              >
                {switchableArtifacts.map((item) => (
                  <option key={item.id} value={item.id}>{item.kind} · {item.path}</option>
                ))}
                {switchableArtifacts.length === 0 ? <option value="">暂无 ONNX / engine ready 产物</option> : null}
              </select>
              <button
                className="console-button primary console-full-button"
                disabled={busy === "model.switch" || selectedModelProjectId === "" || selectedSwitchArtifact === null}
                onClick={switchModel}
                type="button"
              >
                {busy === "model.switch" ? "安全切换中..." : "安全切换模型"}
              </button>
              <label>推理后端</label>
              <select value={readString(runtime?.inference?.selected, "auto")} disabled>
                <option value={readString(runtime?.inference?.selected, "auto")}>
                  {readString(runtime?.inference?.selected, "按模型后缀自动选择")}
                </option>
              </select>
              <label>输入尺寸</label>
              <select value={version?.input_shape ?? ""} disabled>
                <option>{version?.input_shape ?? "模型未发布"} · 当前 {artifact?.path ?? activeModelName}</option>
              </select>
              <label>置信度阈值</label>
              <div className="console-row">
                <input type="range" min="0" max="1" step=".01" value={confidence} onChange={(event) => void updateConfigField("inference", "confidence_threshold", Number(event.target.value))} />
                <input value={confidence.toFixed(2)} readOnly />
              </div>
              <label>NMS 阈值</label>
              <div className="console-row">
                <input type="range" min="0" max="1" step=".01" value={nms} onChange={(event) => void updateConfigField("inference", "nms_threshold", Number(event.target.value))} />
                <input value={nms.toFixed(2)} readOnly />
              </div>
              <label>检测类别</label>
              <select
                value={activeDetectionProfile}
                onChange={(event) => void updateConfigField("inference", "detection_class_profile", event.target.value)}
              >
                {(detectionProfileNames.length > 0 ? detectionProfileNames : ["default"]).map((name) => (
                  <option key={name} value={name}>{name}</option>
                ))}
              </select>
              <div className="console-class-list">
                <button
                  className={activeDetectionClass === "all" ? "console-class-row active" : "console-class-row"}
                  onClick={() => void updateConfigField("inference", "detection_class_filter", "all")}
                  type="button"
                >
                  全部类别
                </button>
                {detectionClasses.map((item, index) => (
                  <button
                    className={activeDetectionClass === String(index) ? "console-class-row active" : "console-class-row"}
                    key={`${index}-${item}`}
                    onClick={() => void updateConfigField("inference", "detection_class_filter", String(index))}
                    type="button"
                  >
                    {item}
                  </button>
                ))}
              </div>
            </div>
            <div className="console-card">
              <h2 className="console-title">推理输出</h2>
              <PreviewFrame runtime={runtime} roiSize={roiSize} />
              <div className="console-kv">
                <span>当前类别</span><b>{readString(target.class_name, "-")}</b>
                <span>最高置信度</span><b>{target.score ? Number(target.score).toFixed(2) : "-"}</b>
                <span>候选框数量</span><b>{detections}</b>
              </div>
            </div>
          </div>
        </section>

        <section className={activePage === "params" ? "console-page active" : "console-page"}>
          <div className="console-metrics">
            <Metric title="控制策略" value={readString(controlConfig.strategy, "pid")} small="strategy" />
            <Metric title="Kp X" value={pidKpX.toFixed(2)} small="axis x" />
            <Metric title="Kp Y" value={pidKpY.toFixed(2)} small="axis y" />
            <Metric title="控制量上限" value={`${kpXMoveMax}/${kpYMoveMax}`} small="x/y" />
          </div>
          <div className="console-grid2">
            <div className="console-card">
              <h2 className="console-title">鼠标移动算法</h2>
              <label>算法模式</label>
              <select
                value={readString(controlConfig.strategy, "pid")}
                onChange={(event) => void updateConfigField("control", "strategy", event.target.value)}
              >
                <option value="pid">PID 平滑追踪</option>
                <option value="proportional">比例速度</option>
                <option value="predictive">预测追踪</option>
              </select>
              <NumberControl label="kp_x" value={pidKpX} min={0} max={2} step={0.01} onCommit={(value) => updateConfigField("control", "pid_kp_x", value)} />
              <NumberControl label="kp_y" value={pidKpY} min={0} max={2} step={0.01} onCommit={(value) => updateConfigField("control", "pid_kp_y", value)} />
              <NumberControl label="ki" value={pidKi} min={0} max={1} step={0.01} onCommit={(value) => updateConfigField("control", "pid_ki", value)} />
              <NumberControl label="kd" value={pidKd} min={0} max={1} step={0.01} onCommit={(value) => updateConfigField("control", "pid_kd", value)} />
              <NumberControl label="kp_x_move_max" value={kpXMoveMax} min={0} max={500} step={1} onCommit={(value) => updateConfigField("control", "kp_x_move_max", value)} />
              <NumberControl label="kp_y_move_max" value={kpYMoveMax} min={0} max={500} step={1} onCommit={(value) => updateConfigField("control", "kp_y_move_max", value)} />
            </div>
            <div className="console-card">
              <h2 className="console-title">控制量反馈</h2>
              <div className="console-kv">
                <span>当前目标</span><b>{readString(target.class_name, "-")}</b>
                <span>dx</span><b>{formatNumber(asRecord(vision.control).dx, 1)}</b>
                <span>dy</span><b>{formatNumber(asRecord(vision.control).dy, 1)}</b>
                <span>输出状态</span><b>{asRecord(vision.control).will_emit === true ? "允许输出" : "等待触发"}</b>
              </div>
            </div>
          </div>
        </section>

        <section className={activePage === "stats" ? "console-page active" : "console-page"}>
          <div className="console-metrics">
            <Metric title="总 FPS" value={formatNumber(statistics?.capture_fps, 1)} small="FPS" />
            <Metric title="平均延迟" value={formatNumber(statistics?.e2e_latency, 1)} small="ms" />
            <Metric title="P95 延迟" value="待机" small="ms" />
            <Metric title="运行时长" value={runtime?.running ? "运行中" : "00:00"} small="time" />
          </div>
          <div className="console-grid3">
            <KvCard title="采集统计" rows={[["成功帧", String(statistics?.capture_counter ?? 0)], ["丢弃帧", String(statistics?.dropped_counter ?? 0)], ["抖动", formatNumber(capture?.frame_period_ms, 2)]]} />
            <KvCard title="推理统计" rows={[["完成帧", String(statistics?.inference_counter ?? 0)], ["平均耗时", formatNumber(statistics?.e2e_latency, 1)], ["最大耗时", "-"]]} />
            <KvCard title="系统状态" rows={[["CPU", "待机"], ["GPU", "待机"], ["温度", "-"]]} />
          </div>
          <div className="console-card">
            <h2 className="console-title">性能占比</h2>
            <Bar label="采集" width={35} />
            <Bar label="预处理" width={18} />
            <Bar label="推理" width={52} />
            <Bar label="后处理" width={24} />
          </div>
        </section>

        <section className={activePage === "latency" ? "console-page active" : "console-page"}>
          <div className="console-metrics">
            <Metric title="采集等待" value={formatNumber(capture?.capture_wait_ms, 2)} small="ms" />
            <Metric title="帧间隔" value={formatNumber(capture?.frame_period_ms, 2)} small="ms" />
            <Metric title="端到端" value={formatNumber(statistics?.e2e_latency, 1)} small="ms" />
            <Metric title="队列积压" value={String(readNumber(asRecord(pipeline.queue).size, 0))} small="frames" />
          </div>
          <div className="console-grid2">
            <div className="console-card">
              <h2 className="console-title">延迟链路</h2>
              <div className="console-timeline">
                <Event label="Capture" value={formatNumber(capture?.capture_wait_ms, 2)} width={30} />
                <Event label="Decode" value="--" width={22} />
                <Event label="Preprocess" value="--" width={18} />
                <Event label="Inference" value={formatNumber(statistics?.e2e_latency, 1)} width={56} />
                <Event label="Postprocess" value="--" width={20} />
              </div>
            </div>
            <KvCard title="采集诊断" rows={[["状态判断", capture?.available ? "采集中" : "等待数据"], ["建议", capture?.available ? "观察丢帧和帧间隔" : "启动后分析"], ["峰值延迟", "-"]]} />
          </div>
        </section>
      </main>
    </section>
  );
}

function NumberControl({
  label,
  value,
  min,
  max,
  step,
  onCommit
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onCommit: (value: number) => Promise<void> | void;
}) {
  return (
    <>
      <label>{label}</label>
      <div className="console-row">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(event) => void onCommit(Number(event.target.value))}
        />
        <input
          type="number"
          min={min}
          max={max}
          step={step}
          value={Number.isInteger(value) ? String(value) : value.toFixed(2)}
          onChange={(event) => {
            const next = Number(event.target.value);
            if (Number.isFinite(next)) {
              void onCommit(next);
            }
          }}
        />
      </div>
    </>
  );
}

function Metric({ title, value, small }: { title: string; value: string; small: string }) {
  return <div className="console-metric">{title}<br />{value}<small>{small}</small></div>;
}

function PreviewCard({ runtime, title, roiSize }: { runtime: RuntimeState | null; title: string; roiSize: number }) {
  return (
    <div className="console-card">
      <h2 className="console-title">{title}</h2>
      <PreviewFrame runtime={runtime} roiSize={roiSize} />
    </div>
  );
}

function PreviewFrame({ runtime, roiSize }: { runtime: RuntimeState | null; roiSize: number }) {
  const configVersion = typeof runtime?.config?.version === "number" ? runtime.config.version : 0;
  return (
    <div className="console-preview" style={{ "--roi-size": `${roiSize}px` } as React.CSSProperties}>
      {runtime?.capture?.available ? <img alt="实时画面 / ROI" src={streamUrl(configVersion, configVersion)} /> : null}
    </div>
  );
}

function KvCard({ title, rows }: { title: string; rows: [string, string][] }) {
  return (
    <div className="console-card">
      <h2 className="console-title">{title}</h2>
      <div className="console-kv">
        {rows.map(([key, value]) => (
          <Fragment key={key}>
            <span>{key}</span>
            <b>{value}</b>
          </Fragment>
        ))}
      </div>
    </div>
  );
}

function Bar({ label, width }: { label: string; width: number }) {
  return (
    <>
      <label>{label}</label>
      <div className="console-bar"><i style={{ width: `${width}%` }} /></div>
    </>
  );
}

function Event({ label, value, width }: { label: string; value: string; width: number }) {
  return (
    <div className="console-event">
      <span>{label}</span>
      <div className="console-bar"><i style={{ width: `${width}%` }} /></div>
      <b>{value} ms</b>
    </div>
  );
}
