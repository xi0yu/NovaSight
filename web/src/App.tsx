import { useCallback, useEffect, useMemo, useState } from "react";

import { StudioShell } from "./app/StudioShell";
import { type StudioViewId } from "./app/navigation";
import {
  EmptyState,
  InlineError,
  LoadingSkeleton,
  Panel,
  StatusIndicator
} from "./components/ui";
import {
  ActiveModel,
  ApiError,
  ConfigFieldSchema,
  ConfigSchemaResponse,
  CaptureCapabilitiesResponse,
  CaptureCapability,
  CaptureSelectPayload,
  CaptureState,
  ExecutorStatus,
  HealthResponse,
  LicenseStatus,
  ModelProject,
  PluginInfo,
  RuntimeConfig,
  RuntimeState,
  TEST_MAX_LICENSE_KEY,
  clearLicenseKey,
  getCaptureCapabilities,
  getConfigSchema,
  getHealth,
  getLicenseStatus,
  getModelProjects,
  getPlugins,
  getRuntimeState,
  saveLicenseKey,
  selectCaptureProfile,
  startInferenceControl,
  statusWebSocketUrl,
  stopCapture,
  stopInferenceControl,
  streamUrl,
  updateRuntimeConfig
} from "./api";

type ErrorKey = "health" | "runtime" | "plugins" | "projects" | "capture";

type LoadState = {
  loading: boolean;
  errors: Partial<Record<ErrorKey, string>>;
  health: HealthResponse | null;
  runtime: RuntimeState | null;
  plugins: PluginInfo[];
  projects: ModelProject[];
  lastUpdated: Date | null;
};

type CapabilityChoice = {
  pixel_format: string;
  width: number;
  height: number;
  fps: number;
};

type CapabilityGroup = {
  pixel_format: string;
  choices: CapabilityChoice[];
};

type ConfigValue = string | number | boolean | null | ConfigValue[] | { [key: string]: ConfigValue };

const LICENSE_CACHE_KEY = "novasight.license.valid";

const initialState: LoadState = {
  loading: true,
  errors: {},
  health: null,
  runtime: null,
  plugins: [],
  projects: [],
  lastUpdated: null
};

function getErrorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    return `${error.status} ${error.message}`;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return "无法连接 NovaSight 后端";
}

function withoutError(
  errors: LoadState["errors"],
  key: ErrorKey
): LoadState["errors"] {
  const next = { ...errors };
  delete next[key];
  return next;
}

function formatTime(date: Date | null): string {
  if (!date) {
    return "从未更新";
  }
  return new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit"
  }).format(date);
}

function formatEpoch(seconds: number | null | undefined): string {
  if (!seconds) {
    return "无";
  }
  return new Intl.DateTimeFormat("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit"
  }).format(new Date(seconds * 1000));
}

function statusTone(value: boolean | undefined): "good" | "warn" | "bad" {
  if (value === true) {
    return "good";
  }
  if (value === false) {
    return "bad";
  }
  return "warn";
}

function formatProfile(capture: CaptureState | undefined): string {
  if (!capture?.profile) {
    return "未配置";
  }
  return `${capture.profile.pixel_format} ${capture.profile.width}x${capture.profile.height} @ ${capture.profile.fps}`;
}

function Field({
  label,
  value,
  mono = false
}: {
  label: string;
  value: React.ReactNode;
  mono?: boolean;
}) {
  return (
    <div className="field">
      <span className="field-label">{label}</span>
      <span className={mono ? "field-value mono" : "field-value"}>{value}</span>
    </div>
  );
}

function OverviewView({
  health,
  runtime,
  loading,
  errors,
  onRefresh,
  onInferenceControlCommand,
  runtimeCommandBusy
}: {
  health: HealthResponse | null;
  runtime: RuntimeState | null;
  loading: boolean;
  errors: Pick<LoadState["errors"], "health" | "runtime">;
  onRefresh: () => void;
  onInferenceControlCommand: (action: "start" | "stop") => void;
  runtimeCommandBusy: boolean;
}) {
  const executor = runtime?.executor;
  const selectedExecutor = executor?.selected;
  const selectedAvailability = selectedExecutor
    ? executor?.executors[selectedExecutor]?.available
    : undefined;

  return (
    <div className="view-grid dashboard-grid">
      <Panel
        title="运行总览"
        eyebrow="核心状态"
        action={
          <div className="panel-actions">
            <InferenceControl
              busy={runtimeCommandBusy}
              running={Boolean(runtime?.running)}
              onCommand={onInferenceControlCommand}
            />
            <button className="button" type="button" onClick={onRefresh}>
              刷新
            </button>
          </div>
        }
      >
        <InlineError message={errors.health ?? errors.runtime} />
        <div className="metric-grid">
          <div className="metric">
            <span>后端</span>
            <StatusIndicator tone={statusTone(health?.ok)}>
              {loading ? "检查中" : health?.ok ? "已连接" : "离线"}
            </StatusIndicator>
          </div>
          <div className="metric">
            <span>运行态</span>
            <StatusIndicator tone={runtime?.running ? "good" : "idle"}>
              {runtime?.running ? "运行中" : "待机"}
            </StatusIndicator>
          </div>
          <div className="metric">
            <span>采集配置</span>
            <strong>{formatProfile(runtime?.capture)}</strong>
          </div>
          <div className="metric">
            <span>控制输出</span>
            <StatusIndicator tone={statusTone(selectedAvailability)}>
              {selectedExecutor ?? "未选择"}
            </StatusIndicator>
          </div>
        </div>
        <div className="field-grid">
          <Field label="采集设备" value={runtime?.capture?.device ?? "/dev/video0"} mono />
          <Field label="采集后端" value={runtime?.capture?.backend ?? "未打开"} mono />
          <Field label="当前模型" value={runtime?.active_model?.project?.name ?? "未发布模型"} />
          <Field
            label="模型文件"
            value={runtime?.active_model?.artifact?.path ?? "未绑定"}
            mono
          />
        </div>
      </Panel>

      <Panel title="输出执行器" eyebrow="可用性">
        <InlineError message={errors.runtime} />
        <ExecutorTable executor={executor} />
      </Panel>
    </div>
  );
}

function ExecutorTable({ executor }: { executor: ExecutorStatus | undefined }) {
  const entries = Object.entries(executor?.executors ?? {});

  if (entries.length === 0) {
    return <EmptyState title="没有执行器状态" detail="后端没有返回控制输出执行器。" />;
  }

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>执行器</th>
            <th>当前选择</th>
            <th>可用</th>
          </tr>
        </thead>
        <tbody>
          {entries.map(([id, state]) => (
            <tr key={id}>
              <td className="mono">{id}</td>
              <td>{executor?.selected === id ? "是" : "否"}</td>
              <td>
                <StatusIndicator tone={state.available ? "good" : "bad"}>
                  {state.available ? "可用" : "不可用"}
                </StatusIndicator>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function CaptureWorkbench({
  runtime,
  error,
  onRuntimeRefresh,
  onInferenceControlCommand,
  runtimeCommandBusy
}: {
  runtime: RuntimeState | null;
  error: string | undefined;
  onRuntimeRefresh: () => Promise<void>;
  onInferenceControlCommand: (action: "start" | "stop") => void;
  runtimeCommandBusy: boolean;
}) {
  const [device, setDevice] = useState(runtime?.capture?.device ?? "/dev/video0");
  const [capabilities, setCapabilities] = useState<CaptureCapabilitiesResponse | null>(null);
  const [captureError, setCaptureError] = useState<string | undefined>(error);
  const [loadingCaps, setLoadingCaps] = useState(false);
  const [applying, setApplying] = useState<string | null>(null);
  const [stoppingCapture, setStoppingCapture] = useState(false);
  const [streamKey, setStreamKey] = useState(Date.now());

  useEffect(() => {
    if (runtime?.capture?.device) {
      setDevice(runtime.capture.device);
    }
  }, [runtime?.capture?.device]);

  useEffect(() => {
    setCaptureError(error);
  }, [error]);

  const groups = useMemo(() => groupCapabilities(capabilities?.capabilities ?? []), [capabilities]);

  const refreshCapabilities = useCallback(async () => {
    setLoadingCaps(true);
    setCaptureError(undefined);
    try {
      const result = await getCaptureCapabilities(device);
      setCapabilities(result);
      if (!result.available) {
        setCaptureError(result.reason || "设备不可用");
      }
    } catch (err) {
      setCaptureError(getErrorMessage(err));
    } finally {
      setLoadingCaps(false);
    }
  }, [device]);

  useEffect(() => {
    void refreshCapabilities();
  }, [refreshCapabilities]);

  const applySelection = useCallback(
    async (payload: CaptureSelectPayload, label: string) => {
      setApplying(label);
      setCaptureError(undefined);
      try {
        await selectCaptureProfile(payload);
        setStreamKey(Date.now());
        await onRuntimeRefresh();
      } catch (err) {
        setCaptureError(`切换失败，已保留上一组可用配置：${getErrorMessage(err)}`);
        await onRuntimeRefresh();
      } finally {
        setApplying(null);
      }
    },
    [onRuntimeRefresh]
  );

  const stopCaptureSession = useCallback(async () => {
    setStoppingCapture(true);
    setCaptureError(undefined);
    try {
      await stopCapture();
      setStreamKey(Date.now());
      await onRuntimeRefresh();
    } catch (err) {
      setCaptureError(getErrorMessage(err));
      await onRuntimeRefresh();
    } finally {
      setStoppingCapture(false);
    }
  }, [onRuntimeRefresh]);

  const applyPreference = (preference: CaptureSelectPayload["preference"], label: string) =>
    applySelection({ device, preference }, label);

  const capture = runtime?.capture;
  const runtimeRunning = Boolean(runtime?.running);

  return (
    <div className="capture-workbench">
      <Panel
        title="采集工作台"
        eyebrow="设备能力与配置切换"
        action={
          <div className="panel-actions">
            <InferenceControl
              busy={runtimeCommandBusy}
              running={runtimeRunning}
              onCommand={onInferenceControlCommand}
            />
            <button
              className="button"
              disabled={stoppingCapture || !capture?.available}
              onClick={() => void stopCaptureSession()}
              type="button"
            >
              {stoppingCapture ? "停止中" : "停止采集"}
            </button>
            <button className="button" type="button" onClick={refreshCapabilities}>
              {loadingCaps ? "读取中" : "刷新能力"}
            </button>
          </div>
        }
      >
        <InlineError message={captureError} />
        {capture?.available ? (
          <div className="inline-note">
            采集会话已打开。浏览器预览按目标帧率取样，推理控制读取同一个最新帧队列。
          </div>
        ) : (
          <div className="inline-note">
            请选择格式并应用，后端会打开正式采集会话；预览不会单独占用采集卡。
          </div>
        )}
        <div className="capture-toolbar">
          <label className="device-input">
            <span>设备</span>
            <input value={device} onChange={(event) => setDevice(event.target.value)} />
          </label>
          <div className="preference-actions" aria-label="推荐配置">
            <button
              className="button"
              disabled={applying !== null}
              onClick={() => applyPreference("auto_high_fps", "高帧率")}
              type="button"
            >
              高帧率
            </button>
            <button
              className="button"
              disabled={applying !== null}
              onClick={() => applyPreference("auto_low_latency", "低延迟")}
              type="button"
            >
              低延迟
            </button>
            <button
              className="button"
              disabled={applying !== null}
              onClick={() => applyPreference("auto_balanced", "均衡")}
              type="button"
            >
              均衡
            </button>
          </div>
        </div>
        <CapabilityTable
          applying={applying}
          groups={groups}
          onApply={(row) =>
            applySelection(
              {
                device,
                preference: "manual",
                pixel_format: row.pixel_format,
                width: row.width,
                height: row.height,
                fps: row.fps
              },
              `${row.pixel_format}-${row.width}-${row.height}-${row.fps}`
            )
          }
        />
      </Panel>

      <Panel title="实时预览" eyebrow="MJPEG 采集流">
        <div className="video-shell live-preview">
          <img
            alt="实时采集画面"
            src={streamUrl(streamKey)}
          />
          <div className="scan-lines" />
          <div className="reticle" />
          <div className="corner-frame corner-frame-tl" />
          <div className="corner-frame corner-frame-tr" />
          <div className="corner-frame corner-frame-bl" />
          <div className="corner-frame corner-frame-br" />
          <div className="video-status">
            <StatusIndicator tone={capture?.available ? "good" : "idle"}>
              {capture?.available ? "采集中" : "未打开"}
            </StatusIndicator>
            <span className="mono">{formatProfile(capture)}</span>
          </div>
          <div className="preview-caption">预览限速输出，采集与推理控制不依赖浏览器帧率</div>
        </div>
      </Panel>

      <Panel title="当前配置" eyebrow="采集状态">
        <CaptureDiagnostics capture={capture} />
      </Panel>
    </div>
  );
}

function InferenceControl({
  running,
  busy,
  onCommand
}: {
  running: boolean;
  busy: boolean;
  onCommand: (action: "start" | "stop") => void;
}) {
  return (
    <button
      className={running ? "button danger-button" : "button primary-button"}
      disabled={busy}
      onClick={() => onCommand(running ? "stop" : "start")}
      type="button"
    >
      {busy ? "处理中" : running ? "停止推理控制" : "启动推理控制"}
    </button>
  );
}

function groupCapabilities(caps: CaptureCapability[]): CapabilityGroup[] {
  const grouped = new Map<string, CapabilityChoice[]>();
  caps.forEach((cap) => {
    const pixelFormat = cap.pixel_format.toUpperCase();
    const choices = grouped.get(pixelFormat) ?? [];
    cap.fps_list.forEach((fps) => {
      choices.push({
        pixel_format: pixelFormat,
        width: cap.width,
        height: cap.height,
        fps
      });
    });
    grouped.set(pixelFormat, choices);
  });
  const order = ["MJPG", "NV12", "YUYV", "BGR3"];
  return Array.from(grouped.entries())
    .map(([pixel_format, choices]) => ({
      pixel_format,
      choices: choices.sort((left, right) =>
        right.fps - left.fps || right.width * right.height - left.width * left.height
      )
    }))
    .sort((left, right) => {
      const leftRank = order.includes(left.pixel_format) ? order.indexOf(left.pixel_format) : 99;
      const rightRank = order.includes(right.pixel_format) ? order.indexOf(right.pixel_format) : 99;
      return leftRank - rightRank || left.pixel_format.localeCompare(right.pixel_format);
    });
}

function CapabilityTable({
  groups,
  applying,
  disabled,
  onApply
}: {
  groups: CapabilityGroup[];
  applying: string | null;
  disabled?: boolean;
  onApply: (row: CapabilityChoice) => void;
}) {
  if (groups.length === 0) {
    return (
      <EmptyState
        title="尚未读取采集能力"
        detail="点击刷新能力，读取 /dev/video0 支持的格式、分辨率和帧率。"
        command="python3 -m novasight doctor camera --device /dev/video0"
      />
    );
  }

  return (
    <div className="capability-groups">
      {groups.map((group, index) => (
        <details className="capability-group" key={group.pixel_format} open={index < 2}>
          <summary>
            <span className="mono">{group.pixel_format}</span>
            <span>{group.choices.length} 组配置</span>
            <strong>{getFormatSummary(group)}</strong>
          </summary>
          <div className="table-wrap capability-table">
            <table>
              <thead>
                <tr>
                  <th>分辨率</th>
                  <th>帧率</th>
                  <th>建议</th>
                  <th>操作</th>
                </tr>
              </thead>
              <tbody>
                {group.choices.map((row) => {
                  const id = `${row.pixel_format}-${row.width}-${row.height}-${row.fps}`;
                  return (
                    <tr key={id}>
                      <td className="mono">
                        {row.width}x{row.height}
                      </td>
                      <td className="mono">{row.fps}</td>
                      <td>
                        <span className={`hint-chip ${getCapabilityTone(row)}`}>
                          {getCapabilityHint(row)}
                        </span>
                      </td>
                      <td>
                        <button
                          className="button compact-button"
                          disabled={applying !== null || disabled}
                          onClick={() => onApply(row)}
                          type="button"
                        >
                          {applying === id ? "应用中" : "应用并启动"}
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </details>
      ))}
    </div>
  );
}

function getFormatSummary(group: CapabilityGroup): string {
  const maxFps = Math.max(...group.choices.map((choice) => choice.fps));
  const maxPixels = Math.max(...group.choices.map((choice) => choice.width * choice.height));
  const maxChoice = group.choices.find((choice) => choice.width * choice.height === maxPixels);
  return `${maxChoice?.width ?? 0}x${maxChoice?.height ?? 0} / ${maxFps}fps`;
}

function getCapabilityHint(row: CapabilityChoice): string {
  if (row.pixel_format === "MJPG" && row.fps >= 120) {
    return "高帧率";
  }
  if (row.pixel_format === "NV12" && row.fps >= 60) {
    return "低延迟";
  }
  if (row.width === 1920 && row.height === 1080) {
    return "均衡";
  }
  return "手动";
}

function getCapabilityTone(row: CapabilityChoice): string {
  if (row.pixel_format === "MJPG" && row.fps >= 120) {
    return "prime";
  }
  if (row.pixel_format === "NV12" && row.fps >= 60) {
    return "cool";
  }
  if (row.width === 1920 && row.height === 1080) {
    return "balanced";
  }
  return "plain";
}

function CaptureDiagnostics({ capture }: { capture: CaptureState | undefined }) {
  if (!capture) {
    return <EmptyState title="没有采集状态" detail="后端没有返回摄像头诊断信息。" />;
  }

  return (
    <dl className="metric-list">
      <div>
        <dt>设备</dt>
        <dd className="mono">{capture.device}</dd>
      </div>
      <div>
        <dt>后端</dt>
        <dd className="mono">{capture.backend ?? "未选择"}</dd>
      </div>
      <div>
        <dt>配置</dt>
        <dd>{formatProfile(capture)}</dd>
      </div>
      <div>
        <dt>采集帧率</dt>
        <dd className="mono">{capture.fps_capture.toFixed(1)} fps</dd>
      </div>
      <div>
        <dt>预览目标</dt>
        <dd className="mono">{capture.preview_target_fps} fps</dd>
      </div>
      <div>
        <dt>预览输出</dt>
        <dd className="mono">{capture.preview_fps.toFixed(1)} fps</dd>
      </div>
      <div>
        <dt>帧间隔</dt>
        <dd className="mono">{capture.frame_period_ms.toFixed(2)} ms</dd>
      </div>
      <div>
        <dt>读取等待</dt>
        <dd className="mono">{capture.capture_wait_ms.toFixed(2)} ms</dd>
      </div>
      <div>
        <dt>丢帧</dt>
        <dd className="mono">{capture.frames_dropped}</dd>
      </div>
      <div>
        <dt>预览帧</dt>
        <dd className="mono">{capture.preview_output_frames}</dd>
      </div>
      <div>
        <dt>预览丢帧</dt>
        <dd className="mono">{capture.preview_dropped}</dd>
      </div>
      <div>
        <dt>恢复次数</dt>
        <dd className="mono">{capture.recoveries}</dd>
      </div>
      <div>
        <dt>最近错误</dt>
        <dd>{capture.last_error ?? "无"}</dd>
      </div>
    </dl>
  );
}

function ModelsView({
  projects,
  activeModel,
  error
}: {
  projects: ModelProject[];
  activeModel: ActiveModel | null;
  error: string | undefined;
}) {
  return (
    <div className="view-grid models-grid">
      <Panel title="当前模型" eyebrow="运行绑定">
        {activeModel ? (
          <div className="field-grid">
            <Field label="项目" value={activeModel.project?.name ?? "未知项目"} />
            <Field label="部署编号" value={activeModel.deployment.id} mono />
            <Field label="模型文件" value={activeModel.artifact?.path ?? "缺少文件"} mono />
            <Field label="文件状态" value={activeModel.artifact?.status ?? "未知"} />
          </div>
        ) : (
          <EmptyState
            title="没有发布模型"
            detail="发布可用模型后，推理运行时会在这里显示绑定状态。"
            command="POST /api/models/projects/{project_id}/publish"
          />
        )}
      </Panel>
      <Panel title="模型项目" eyebrow="注册表">
        <InlineError message={error} />
        {projects.length > 0 ? (
          <div className="project-list">
            {projects.map((project) => (
              <article className="project-row" key={project.id}>
                <div>
                  <strong>{project.name}</strong>
                  <span>{project.description || "没有描述"}</span>
                </div>
                <code>#{project.id}</code>
              </article>
            ))}
          </div>
        ) : (
          <EmptyState
            title="模型注册表为空"
            detail="创建模型项目、添加版本、转换产物并发布后，这里会展示项目。"
          />
        )}
      </Panel>
    </div>
  );
}

function PluginsView({
  plugins,
  error
}: {
  plugins: PluginInfo[];
  error: string | undefined;
}) {
  const groups = useMemo(
    () => ({
      vision: plugins.filter((plugin) => plugin.kind === "vision"),
      control: plugins.filter((plugin) => plugin.kind === "control"),
      other: plugins.filter((plugin) => plugin.kind !== "vision" && plugin.kind !== "control")
    }),
    [plugins]
  );

  if (plugins.length === 0) {
    return (
      <Panel title="插件" eyebrow="算法链">
        <InlineError message={error} />
        <EmptyState title="没有加载插件" detail="后端返回的插件链为空。" />
      </Panel>
    );
  }

  return (
    <div className="view-grid plugins-grid">
      {error ? (
        <Panel title="插件请求" eyebrow="数据质量">
          <InlineError message={error} />
        </Panel>
      ) : null}
      <PluginGroup title="视觉分析" plugins={groups.vision} />
      <PluginGroup title="控制算法" plugins={groups.control} />
      {groups.other.length > 0 ? <PluginGroup title="其他" plugins={groups.other} /> : null}
    </div>
  );
}

function PluginGroup({ title, plugins }: { title: string; plugins: PluginInfo[] }) {
  return (
    <Panel title={`${title}插件`} eyebrow="已加载模块">
      {plugins.length > 0 ? (
        <div className="plugin-list">
          {plugins.map((plugin) => (
            <article className="plugin-row" key={plugin.plugin_id}>
              <div>
                <strong className="mono">{plugin.plugin_id}</strong>
                <span>{plugin.kind}</span>
              </div>
              <StatusIndicator tone={plugin.enabled ? "good" : "idle"}>
                {plugin.enabled ? "启用" : "停用"}
              </StatusIndicator>
            </article>
          ))}
        </div>
      ) : (
        <EmptyState title={`没有${title}插件`} detail="这个链路暂时没有模块。" />
      )}
    </Panel>
  );
}

function getConfigValue(config: RuntimeConfig, path: string): ConfigValue {
  return path.split(".").reduce<ConfigValue>((current, part) => {
    if (typeof current === "object" && current !== null && !Array.isArray(current)) {
      return current[part] ?? "";
    }
    return "";
  }, config);
}

function setConfigValue(
  config: RuntimeConfig,
  path: string,
  rawValue: string,
  field: ConfigFieldSchema
): RuntimeConfig {
  const current = getConfigValue(config, path);
  const value =
    field.type === "int"
      ? Number.parseInt(rawValue || "0", 10)
      : field.type === "float"
        ? Number.parseFloat(rawValue || "0")
        : field.type === "select" && typeof current === "number"
          ? Number.parseInt(rawValue || "0", 10)
          : rawValue;
  const next = structuredClone(config);
  const parts = path.split(".");
  let cursor: Record<string, ConfigValue> = next;
  for (const part of parts.slice(0, -1)) {
    const child = cursor[part];
    if (typeof child !== "object" || child === null || Array.isArray(child)) {
      cursor[part] = {};
    }
    cursor = cursor[part] as Record<string, ConfigValue>;
  }
  cursor[parts[parts.length - 1]] = value;
  return next;
}

function LicensePanel({
  license,
  onLicenseChange
}: {
  license: LicenseStatus | null;
  onLicenseChange: (license: LicenseStatus) => void;
}) {
  const [licenseInput, setLicenseInput] = useState("");
  const [message, setMessage] = useState<string | undefined>();
  const [error, setError] = useState<string | undefined>();

  const saveLicense = useCallback(async () => {
    setError(undefined);
    setMessage(undefined);
    try {
      const status = await saveLicenseKey(licenseInput);
      localStorage.setItem(LICENSE_CACHE_KEY, status.valid ? "1" : "0");
      onLicenseChange(status);
      setLicenseInput("");
      setMessage("卡密已激活，界面不回显明文。");
    } catch (err) {
      setError(getErrorMessage(err));
    }
  }, [licenseInput, onLicenseChange]);

  const clearLicense = useCallback(async () => {
    setError(undefined);
    setMessage(undefined);
    try {
      const status = await clearLicenseKey();
      localStorage.removeItem(LICENSE_CACHE_KEY);
      onLicenseChange(status);
      setMessage("卡密已清除。");
    } catch (err) {
      setError(getErrorMessage(err));
    }
  }, [onLicenseChange]);

  return (
    <div className="license-panel">
      <InlineError message={error} />
      {message ? <div className="inline-note">{message}</div> : null}
      <StatusIndicator tone={license?.valid ? "good" : "idle"}>
        {license?.valid ? "授权有效" : "未授权"}
      </StatusIndicator>
      <div className="field-grid">
        <Field label="授权等级" value={license?.tier || "无"} mono />
        <Field label="指纹" value={license?.fingerprint || "无"} mono />
        <Field label="创建时间" value={formatEpoch(license?.created_at)} />
        <Field label="激活时间" value={formatEpoch(license?.activated_at)} />
        <Field label="到期时间" value={formatEpoch(license?.expires_at)} />
        <Field
          label="期限"
          value={
            license?.duration_value
              ? `${license.duration_value} ${license.duration_unit}`
              : "无"
          }
        />
      </div>
      <label className="config-field">
        <span>卡密</span>
        <input
          type="password"
          value={licenseInput}
          placeholder="输入授权码，激活后不回显"
          onChange={(event) => setLicenseInput(event.target.value)}
        />
      </label>
      <div className="preference-actions">
        <button className="button" type="button" onClick={saveLicense} disabled={!licenseInput.trim()}>
          激活卡密
        </button>
        <button className="button" type="button" onClick={() => setLicenseInput(TEST_MAX_LICENSE_KEY)}>
          填入测试卡密
        </button>
        <button className="button" type="button" onClick={clearLicense}>
          清除
        </button>
      </div>
      <div className="license-features">
        {(license?.features ?? []).map((feature) => (
          <span key={feature}>{feature}</span>
        ))}
      </div>
    </div>
  );
}

function LicenseGate({
  license,
  loading,
  error,
  onRefresh,
  onLicenseChange
}: {
  license: LicenseStatus | null;
  loading: boolean;
  error: string | undefined;
  onRefresh: () => void;
  onLicenseChange: (license: LicenseStatus) => void;
}) {
  return (
    <main className="app-shell license-shell">
      <section className="license-gate">
        <div className="brand-block">
          <span className="brand-mark"><span>NS</span></span>
          <div>
            <h1>NovaSight</h1>
            <p>请输入卡密后进入 Jetson 实时视觉工作台</p>
          </div>
        </div>
        <InlineError message={error} />
        {loading ? <div className="inline-note">正在校验本机授权状态。</div> : null}
        <LicensePanel license={license} onLicenseChange={onLicenseChange} />
        <button className="button compact-button" type="button" onClick={onRefresh}>
          重新校验
        </button>
      </section>
    </main>
  );
}

function SettingsView({
  runtime,
  license,
  onRuntimeRefresh,
  onLicenseChange
}: {
  runtime: RuntimeState | null;
  license: LicenseStatus | null;
  onRuntimeRefresh: () => Promise<void>;
  onLicenseChange: (license: LicenseStatus) => void;
}) {
  const [schema, setSchema] = useState<ConfigSchemaResponse | null>(null);
  const [config, setConfig] = useState<RuntimeConfig | null>(null);
  const [message, setMessage] = useState<string | undefined>();
  const [error, setError] = useState<string | undefined>();

  const loadSettings = useCallback(async () => {
    setError(undefined);
    try {
      const nextSchema = await getConfigSchema();
      setSchema(nextSchema);
      setConfig(nextSchema.values);
    } catch (err) {
      setError(getErrorMessage(err));
    }
  }, []);

  useEffect(() => {
    void loadSettings();
  }, [loadSettings]);

  const saveConfig = useCallback(async () => {
    if (!config) {
      return;
    }
    setError(undefined);
    setMessage(undefined);
    try {
      const result = await updateRuntimeConfig(config);
      setSchema(result.schema);
      setConfig(result.config);
      setMessage(result.restart_required ? "配置已保存，推理控制需要重启后完全生效。" : "配置已保存并同步到运行态。");
      await onRuntimeRefresh();
    } catch (err) {
      setError(getErrorMessage(err));
    }
  }, [config, onRuntimeRefresh]);

  return (
    <div className="settings-workbench">
      <Panel
        title="运行配置"
        eyebrow="配置与运行同源"
        action={
          <button className="button" type="button" onClick={saveConfig} disabled={!config}>
            保存配置
          </button>
        }
      >
        <InlineError message={error} />
        {message ? <div className="inline-note">{message}</div> : null}
        {schema && config ? (
          <div className="config-sections">
            {schema.sections.map((section) => (
              <section className="config-section" key={section.id}>
                <h3>{section.label}</h3>
                <div className="config-grid">
                  {section.fields.map((field) => {
                    const value = getConfigValue(config, field.path);
                    return (
                      <label className="config-field" key={field.path}>
                        <span>
                          {field.label}
                          {field.restart_required ? <em>需重启链路</em> : null}
                        </span>
                        {field.type === "select" ? (
                          <select
                            value={String(value ?? "")}
                            onChange={(event) =>
                              setConfig(setConfigValue(config, field.path, event.target.value, field))
                            }
                          >
                            {(field.options ?? []).map((option) => (
                              <option key={option} value={option}>
                                {option || "跟随默认"}
                              </option>
                            ))}
                          </select>
                        ) : (
                          <input
                            type={field.type === "string" ? "text" : "number"}
                            min={field.min}
                            max={field.max}
                            step={field.type === "float" ? "0.1" : "1"}
                            value={String(value ?? "")}
                            onChange={(event) =>
                              setConfig(setConfigValue(config, field.path, event.target.value, field))
                            }
                          />
                        )}
                      </label>
                    );
                  })}
                </div>
              </section>
            ))}
          </div>
        ) : (
          <EmptyState title="配置 schema 未加载" detail="后端会返回配置字段、类型、范围和枚举选项。" />
        )}
      </Panel>

      <Panel title="卡密管理" eyebrow="本机授权">
        <LicensePanel license={license} onLicenseChange={onLicenseChange} />
      </Panel>

      <Panel title="运行态对照" eyebrow="实时状态">
        <div className="field-grid">
          <Field label="运行中" value={runtime?.running ? "是" : "否"} />
          <Field label="配置版本" value={String(runtime?.config?.version ?? 0)} mono />
          <Field label="默认执行器" value={runtime?.executor.selected ?? "未加载"} mono />
          <Field label="采集配置" value={formatProfile(runtime?.capture)} />
        </div>
      </Panel>
    </div>
  );
}

function LicenseView({
  license,
  onLicenseChange
}: {
  license: LicenseStatus | null;
  onLicenseChange: (license: LicenseStatus) => void;
}) {
  return (
    <div className="view-grid">
      <Panel title="卡密管理" eyebrow="本机授权">
        <LicensePanel license={license} onLicenseChange={onLicenseChange} />
      </Panel>
    </div>
  );
}

export default function App() {
  const [activeView, setActiveView] = useState<StudioViewId>("devices");
  const [state, setState] = useState<LoadState>(initialState);
  const [runtimeCommandBusy, setRuntimeCommandBusy] = useState(false);
  const [license, setLicense] = useState<LicenseStatus | null>(null);
  const [licenseLoading, setLicenseLoading] = useState(
    localStorage.getItem(LICENSE_CACHE_KEY) === "1"
  );
  const [licenseError, setLicenseError] = useState<string | undefined>();

  const loadLicense = useCallback(async () => {
    setLicenseLoading(true);
    setLicenseError(undefined);
    try {
      const status = await getLicenseStatus();
      setLicense(status);
      if (status.valid) {
        localStorage.setItem(LICENSE_CACHE_KEY, "1");
      } else {
        localStorage.removeItem(LICENSE_CACHE_KEY);
      }
      return status;
    } catch (err) {
      setLicenseError(getErrorMessage(err));
      localStorage.removeItem(LICENSE_CACHE_KEY);
      return null;
    } finally {
      setLicenseLoading(false);
    }
  }, []);

  const handleLicenseChange = useCallback((status: LicenseStatus) => {
    setLicense(status);
    if (status.valid) {
      localStorage.setItem(LICENSE_CACHE_KEY, "1");
    } else {
      localStorage.removeItem(LICENSE_CACHE_KEY);
      setState(initialState);
    }
  }, []);

  const load = useCallback(async () => {
    setState((current) => ({ ...current, loading: true, errors: {} }));
    const [health, runtime, plugins, projects] = await Promise.allSettled([
      getHealth(),
      getRuntimeState(),
      getPlugins(),
      getModelProjects()
    ]);
    setState((current) => {
      const errors: LoadState["errors"] = {};
      if (health.status === "rejected") {
        errors.health = getErrorMessage(health.reason);
      }
      if (runtime.status === "rejected") {
        errors.runtime = getErrorMessage(runtime.reason);
      }
      if (plugins.status === "rejected") {
        errors.plugins = getErrorMessage(plugins.reason);
      }
      if (projects.status === "rejected") {
        errors.projects = getErrorMessage(projects.reason);
      }
      return {
        loading: false,
        errors,
        health: health.status === "fulfilled" ? health.value : current.health,
        runtime: runtime.status === "fulfilled" ? runtime.value : current.runtime,
        plugins: plugins.status === "fulfilled" ? plugins.value : current.plugins,
        projects: projects.status === "fulfilled" ? projects.value : current.projects,
        lastUpdated: new Date()
      };
    });
  }, []);

  useEffect(() => {
    void loadLicense();
  }, [loadLicense]);

  useEffect(() => {
    if (license?.valid) {
      void load();
    }
  }, [license?.valid, load]);

  useEffect(() => {
    if (!license?.valid) {
      return undefined;
    }
    const socket = new WebSocket(statusWebSocketUrl());
    socket.onmessage = (event) => {
      try {
        const runtime = JSON.parse(String(event.data)) as RuntimeState;
        setState((current) => ({
          ...current,
          runtime,
          lastUpdated: new Date()
        }));
      } catch {
        // Ignore malformed status frames; REST refresh still provides recovery.
      }
    };
    return () => socket.close();
  }, [license?.valid]);

  const handleInferenceControlCommand = useCallback(
    async (action: "start" | "stop") => {
      setRuntimeCommandBusy(true);
      setState((current) => ({
        ...current,
        errors: withoutError(current.errors, "runtime")
      }));
      try {
        if (action === "start") {
          await startInferenceControl();
        } else {
          await stopInferenceControl();
        }
        await load();
      } catch (err) {
        setState((current) => ({
          ...current,
          errors: { ...current.errors, runtime: getErrorMessage(err) }
        }));
      } finally {
        setRuntimeCommandBusy(false);
      }
    },
    [load]
  );

  if (!license?.valid) {
    return (
      <LicenseGate
        license={license}
        loading={licenseLoading}
        error={licenseError}
        onRefresh={() => void loadLicense()}
        onLicenseChange={handleLicenseChange}
      />
    );
  }

  const activeModel = state.runtime?.active_model ?? null;
  const hasErrors = Object.keys(state.errors).length > 0;
  const shellCopy: Record<StudioViewId, { title: string; subtitle: string }> = {
    dashboard: {
      title: "Dashboard",
      subtitle: "Track runtime health, pipeline readiness, and the current control path."
    },
    devices: {
      title: "Devices",
      subtitle: "Configure capture profiles, inspect camera capability sets, and monitor preview."
    },
    models: {
      title: "Models",
      subtitle: "Review active deployments and the registered project inventory."
    },
    config: {
      title: "Config",
      subtitle: "Adjust runtime configuration and compare the live state with saved settings."
    },
    plugins: {
      title: "Plugins",
      subtitle: "Inspect loaded vision and control modules in the production chain."
    },
    license: {
      title: "License",
      subtitle: "Manage the local activation key and verify feature entitlement status."
    }
  };
  const shellStatus = (
    <>
      <StatusIndicator tone={hasErrors ? "bad" : state.health?.ok ? "good" : "warn"}>
        {hasErrors ? "部分异常" : state.health?.ok ? "已连接" : "连接中"}
      </StatusIndicator>
      <span className="last-updated">更新于 {formatTime(state.lastUpdated)}</span>
    </>
  );

  return (
    <StudioShell
      activeView={activeView}
      title={shellCopy[activeView].title}
      subtitle={shellCopy[activeView].subtitle}
      status={shellStatus}
      onNavigate={setActiveView}
    >
      {hasErrors ? (
        <div className="alert" role="alert">
          <strong>后端请求异常</strong>
          <span>{Object.values(state.errors).join(" / ")}</span>
          <button className="button compact-button" type="button" onClick={load}>
            重试
          </button>
        </div>
      ) : null}

      {state.loading && !state.runtime ? (
        <LoadingSkeleton />
      ) : null}

      <div className="view-stack">
        {activeView === "dashboard" ? (
          <OverviewView
            health={state.health}
            loading={state.loading}
            runtime={state.runtime}
            errors={state.errors}
            onRefresh={load}
            onInferenceControlCommand={(action) => void handleInferenceControlCommand(action)}
            runtimeCommandBusy={runtimeCommandBusy}
          />
        ) : null}
        {activeView === "devices" ? (
          <CaptureWorkbench
            runtime={state.runtime}
            error={state.errors.capture}
            onRuntimeRefresh={load}
            onInferenceControlCommand={(action) => void handleInferenceControlCommand(action)}
            runtimeCommandBusy={runtimeCommandBusy}
          />
        ) : null}
        {activeView === "models" ? (
          <ModelsView
            projects={state.projects}
            activeModel={activeModel}
            error={state.errors.projects}
          />
        ) : null}
        {activeView === "plugins" ? (
          <PluginsView plugins={state.plugins} error={state.errors.plugins} />
        ) : null}
        {activeView === "config" ? (
          <SettingsView
            runtime={state.runtime}
            license={license}
            onRuntimeRefresh={load}
            onLicenseChange={handleLicenseChange}
          />
        ) : null}
        {activeView === "license" ? (
          <LicenseView license={license} onLicenseChange={handleLicenseChange} />
        ) : null}
      </div>
    </StudioShell>
  );
}
