import { useCallback, useEffect, useState } from "react";

import {
  type ConfigFieldSchema,
  type ConfigSchemaResponse,
  type LicenseStatus,
  type RuntimeConfig,
  type RuntimeState,
  getConfigSchema,
  updateRuntimeConfig
} from "../../api";
import { Badge, EmptyState, InlineError, Panel } from "../../components/ui";
import { LicensePanel } from "../license/LicensePanel";
import { Field } from "../shared/Field";
import { formatProfile, getErrorMessage } from "../shared/format";
import { getRuntimeMainlineStatus } from "../shared/runtimeStatus";

export type ConfigValue =
  | string
  | number
  | boolean
  | null
  | ConfigValue[]
  | { [key: string]: ConfigValue };

const DANGEROUS_CONFIG_PATHS = new Set([
  "capture.device",
  "capture.pixel_format",
  "capture.width",
  "capture.height",
  "capture.fps",
  "hardware.kind",
  "hardware.host",
  "hardware.port",
  "hardware.serial_port"
]);

export function getConfigValue(config: RuntimeConfig, path: string): ConfigValue {
  return path.split(".").reduce<ConfigValue>((current, part) => {
    if (typeof current === "object" && current !== null && !Array.isArray(current)) {
      return current[part] ?? "";
    }
    return "";
  }, config);
}

export function setConfigValue(
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
        : field.type === "bool"
          ? rawValue === "true"
        : field.type === "string_list"
          ? rawValue.split(",").map((item) => item.trim()).filter(Boolean).slice(0, 2)
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

export function ConfigView({
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
  const [initialConfig, setInitialConfig] = useState<RuntimeConfig | null>(null);
  const [message, setMessage] = useState<string | undefined>();
  const [error, setError] = useState<string | undefined>();
  const [lastSyncedAt, setLastSyncedAt] = useState<string>("尚未同步");

  const loadSettings = useCallback(async () => {
    setError(undefined);
    try {
      const nextSchema = await getConfigSchema();
      setSchema(nextSchema);
      setConfig(nextSchema.values);
      setInitialConfig(structuredClone(nextSchema.values));
      setLastSyncedAt(new Date().toLocaleTimeString("zh-CN", { hour12: false }));
    } catch (err) {
      setError(getErrorMessage(err));
    }
  }, []);

  useEffect(() => {
    void loadSettings();
  }, [loadSettings]);

  const changedFields =
    schema && config && initialConfig
      ? schema.sections
          .flatMap((section) => section.fields)
          .filter(
            (field) =>
              JSON.stringify(getConfigValue(initialConfig, field.path)) !==
              JSON.stringify(getConfigValue(config, field.path))
          )
      : [];
  const changedPaths = changedFields.map((field) => field.path);
  const restartImpactedPaths = changedFields
    .filter((field) => field.restart_required)
    .map((field) => field.path);
  const dangerousPaths = changedFields
    .filter((field) => DANGEROUS_CONFIG_PATHS.has(field.path))
    .map((field) => field.path);
  const isDirty =
    config && initialConfig ? JSON.stringify(initialConfig) !== JSON.stringify(config) : false;
  const canWriteConfig = Boolean(license?.features.includes("config_write"));
  const runtimeMainlineStatus = getRuntimeMainlineStatus(runtime);
  const runtimeSelectedBackend = runtime?.inference?.selected;
  const runtimeMainlineSelected = runtimeSelectedBackend === "nvmm_latest";
  const runtimeStateLabel = runtimeMainlineSelected
    ? runtimeMainlineStatus.failed
      ? "主链故障"
      : runtimeMainlineStatus.running
        ? runtimeMainlineStatus.hasRuntimeConsumption
          ? "主链已消费"
        : runtimeMainlineStatus.hasInferenceSignal
            ? "等待 runtime 消费"
            : "等待自定义推理输出"
        : "主链未启动"
    : runtime?.running
      ? "运行中"
      : "未运行";
  const runtimeStateDetail = runtimeMainlineSelected
    ? runtimeMainlineStatus.failed
      ? runtimeMainlineStatus.failureMessage || "后端报告主链故障。"
      : runtimeMainlineStatus.running
        ? runtimeMainlineStatus.progressSummary
        : "DeepStream采集 + 自定义推理主链未持有采集、推理与控制链路。"
    : runtime?.running
      ? "传统 runtime 线程正在运行。"
      : "传统 runtime 线程未运行。";

  const saveConfig = useCallback(async () => {
    if (!config || !isDirty || !canWriteConfig) {
      return;
    }
    setError(undefined);
    setMessage(undefined);
    try {
      if (restartImpactedPaths.length > 0 || dangerousPaths.length > 0) {
        const warningLines = ["以下配置项已修改：", ...changedPaths.map((path) => `- ${path}`)];
        if (restartImpactedPaths.length > 0) {
          warningLines.push("", `需要重启链路后完全生效：${restartImpactedPaths.join(", ")}`);
        }
        if (dangerousPaths.length > 0) {
          warningLines.push("", `涉及采集或硬件关键项，请确认设备与连接状态：${dangerousPaths.join(", ")}`);
        }
        warningLines.push("", "确认继续保存？");

        if (!window.confirm(warningLines.join("\n"))) {
          return;
        }
      }

      const result = await updateRuntimeConfig(config);
      setSchema(result.schema);
      setConfig(result.config);
      setInitialConfig(structuredClone(result.config));
      setLastSyncedAt(new Date().toLocaleTimeString("zh-CN", { hour12: false }));
      setMessage(result.restart_required ? "配置已保存，相关运行模块重启后完全生效。" : "配置已保存并同步到运行态。");
      await onRuntimeRefresh();
    } catch (err) {
      setError(getErrorMessage(err));
    }
  }, [
    canWriteConfig,
    changedPaths,
    config,
    dangerousPaths,
    isDirty,
    onRuntimeRefresh,
    restartImpactedPaths
  ]);

  return (
    <div className="settings-workbench">
      <Panel
        title="运行配置"
        eyebrow="配置与运行同源"
        action={
          <div className="panel-actions">
            {!isDirty && config ? <div className="panel-action-note">无未保存修改</div> : null}
            <button
              className={`button ${!isDirty ? "config-save-button is-clean" : ""}`.trim()}
              type="button"
              onClick={saveConfig}
              disabled={!config || !isDirty || !canWriteConfig}
            >
              保存配置
            </button>
          </div>
        }
      >
        <InlineError message={error} />
        {message ? <div className="inline-note">{message}</div> : null}
        {!canWriteConfig ? (
          <div className="inline-note">
            当前授权仅允许读取配置。需要写入配置权限后才能修改并保存运行参数。
          </div>
        ) : null}
        {schema && config ? (
          <>
          <div className="config-sync-summary">
            <div>
              <span>同步状态</span>
              <strong>{isDirty ? "有未保存修改" : "已同步运行配置"}</strong>
            </div>
            <div>
              <span>上次同步</span>
              <strong>{lastSyncedAt}</strong>
            </div>
            <div>
              <span>修改项</span>
              <strong>{changedFields.length}</strong>
            </div>
            <div>
              <span>需重启</span>
              <strong>{restartImpactedPaths.length}</strong>
            </div>
            <div>
              <span>关键项</span>
              <strong>{dangerousPaths.length}</strong>
            </div>
          </div>
          <div className="config-sections">
            {schema.sections.map((section) => (
              <section className="config-section" key={section.id}>
                <h3>{section.label}</h3>
                <div className="config-grid">
                  {section.fields.map((field) => {
                    const value = getConfigValue(config, field.path);
                    const isFieldDirty =
                      initialConfig &&
                      JSON.stringify(getConfigValue(initialConfig, field.path)) !== JSON.stringify(value);
                    return (
                      <label className="config-field" key={field.path}>
                        <span>
                          <span className="config-field-label">
                            <span>{field.label}</span>
                            {isFieldDirty ? <Badge tone="idle">已修改</Badge> : null}
                          </span>
                          {field.restart_required ? <em>需重启链路</em> : null}
                        </span>
                        {field.type === "bool" ? (
                          <input
                            type="checkbox"
                            checked={Boolean(value)}
                            disabled={!canWriteConfig}
                            onChange={(event) =>
                              setConfig(setConfigValue(config, field.path, String(event.target.checked), field))
                            }
                          />
                        ) : field.type === "select" ? (
                          <select
                            value={String(value ?? "")}
                            disabled={!canWriteConfig}
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
                            type={field.type === "string" || field.type === "string_list" ? "text" : "number"}
                            min={field.min}
                            max={field.max}
                            step={field.type === "float" ? "0.1" : "1"}
                            value={Array.isArray(value) ? value.join(", ") : String(value ?? "")}
                            disabled={!canWriteConfig}
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
          </>
        ) : (
          <EmptyState title="配置 schema 未加载" detail="后端会返回配置字段、类型、范围和枚举选项。" />
        )}
      </Panel>

      <Panel title="卡密管理" eyebrow="本机授权">
        <LicensePanel license={license} onLicenseChange={onLicenseChange} />
      </Panel>

      <Panel title="运行态对照" eyebrow="实时状态">
        <div className="field-grid">
          <Field label="运行态" value={runtimeStateLabel} />
          <Field label="运行证据" value={runtimeStateDetail} />
          <Field label="配置版本" value={String(runtime?.config?.version ?? 0)} mono />
          <Field label="默认执行器" value={runtime?.executor.selected ?? "未加载"} mono />
          <Field label="采集状态" value={formatProfile(runtime?.capture)} />
        </div>
      </Panel>
    </div>
  );
}
