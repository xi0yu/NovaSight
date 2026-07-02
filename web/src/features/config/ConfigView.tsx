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
import { EmptyState, InlineError, Panel } from "../../components/ui";
import { LicensePanel } from "../license/LicenseView";
import { Field } from "../shared/Field";
import { formatProfile, getErrorMessage } from "../shared/format";

export type ConfigValue =
  | string
  | number
  | boolean
  | null
  | ConfigValue[]
  | { [key: string]: ConfigValue };

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
