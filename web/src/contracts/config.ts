export type RuntimeConfigValue =
  | string
  | number
  | boolean
  | null
  | RuntimeConfigValue[]
  | { [key: string]: RuntimeConfigValue };

export type RuntimeConfig = Record<string, RuntimeConfigValue>;

export type ConfigApplyMode = "hot_update" | "epoch_reload" | "process_restart";

export interface ConfigFieldSchema {
  path: string;
  label: string;
  type: "string" | "int" | "float" | "select" | "bool";
  options?: string[];
  min?: number;
  max?: number;
  unit?: string;
  description?: string;
  apply_mode: ConfigApplyMode;
  restart_required: boolean;
}

export interface ConfigSectionSchema {
  id: string;
  label: string;
  fields: ConfigFieldSchema[];
}

export interface ConfigAlgorithmResponseSchema {
  formula: string;
  radial_multiplier_formula: string;
  atan_scale_counts: number;
}

export interface ConfigAlgorithmPredictionSchema {
  model: string;
  aim_history_points: number;
  velocity_segments: number;
}

export interface ConfigAlgorithmSchema {
  id: string;
  label: string;
  response: ConfigAlgorithmResponseSchema;
  prediction: ConfigAlgorithmPredictionSchema;
}

export interface ConfigSchemaResponse {
  version: number;
  algorithm: ConfigAlgorithmSchema;
  values: RuntimeConfig;
  sections: ConfigSectionSchema[];
}

export interface ConfigUpdateResponse {
  config: RuntimeConfig;
  apply_mode: ConfigApplyMode;
  restart_required: boolean;
  applied: boolean;
  rolled_back: boolean;
  message: string;
}

export type ConfigCommandPayload =
  | {
      command: "set_output_gate";
      enabled: boolean;
      expected_revision?: number;
    }
  | {
      command: "set_trigger_mode";
      mode: "always" | "hardware";
      expected_revision?: number;
    };

export class ConfigContractError extends Error {
  constructor(path: string, expected: string) {
    super(`配置数据契约错误：${path} 应为 ${expected}`);
    this.name = "ConfigContractError";
  }
}

function decodeRuntimeConfigValue(value: unknown, path: string): RuntimeConfigValue {
  if (value === null || typeof value === "string" || typeof value === "boolean") {
    return value;
  }
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new ConfigContractError(path, "finite number");
    return value;
  }
  if (Array.isArray(value)) {
    return value.map((item, index) => decodeRuntimeConfigValue(item, `${path}[${index}]`));
  }
  if (typeof value === "object") {
    const decoded: Record<string, RuntimeConfigValue> = {};
    for (const [key, item] of Object.entries(value)) {
      decoded[key] = decodeRuntimeConfigValue(item, `${path}.${key}`);
    }
    return decoded;
  }
  throw new ConfigContractError(path, "JSON value");
}

function expectRecord(value: unknown, path: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new ConfigContractError(path, "object");
  }
  return value as Record<string, unknown>;
}

function expectString(value: unknown, path: string): string {
  if (typeof value !== "string") throw new ConfigContractError(path, "string");
  return value;
}

function expectBoolean(value: unknown, path: string): boolean {
  if (typeof value !== "boolean") throw new ConfigContractError(path, "boolean");
  return value;
}

function expectFiniteNumber(value: unknown, path: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new ConfigContractError(path, "finite number");
  }
  return value;
}

function expectUnsignedInteger(value: unknown, path: string): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0) {
    throw new ConfigContractError(path, "unsigned safe integer");
  }
  return value;
}

function expectLiteral<T extends string>(
  value: unknown,
  path: string,
  allowed: readonly T[]
): T {
  if (typeof value !== "string" || !(allowed as readonly string[]).includes(value)) {
    throw new ConfigContractError(path, allowed.join(" | "));
  }
  return value as T;
}

function expectOptionalString(value: unknown, path: string): string | undefined {
  return value === undefined ? undefined : expectString(value, path);
}

function expectOptionalNumber(value: unknown, path: string): number | undefined {
  return value === undefined ? undefined : expectFiniteNumber(value, path);
}

function expectOptionalStringArray(value: unknown, path: string): string[] | undefined {
  if (value === undefined) return undefined;
  if (!Array.isArray(value)) throw new ConfigContractError(path, "string[]");
  return value.map((item, index) => expectString(item, `${path}[${index}]`));
}

export function decodeRuntimeConfig(value: unknown): RuntimeConfig {
  const decoded = decodeRuntimeConfigValue(value, "config");
  if (typeof decoded !== "object" || decoded === null || Array.isArray(decoded)) {
    throw new ConfigContractError("config", "object");
  }
  return decoded;
}

export function decodeConfigSchema(value: unknown): ConfigSchemaResponse {
  const record = expectRecord(value, "config_schema");
  const algorithm = expectRecord(record.algorithm, "config_schema.algorithm");
  const response = expectRecord(algorithm.response, "config_schema.algorithm.response");
  const prediction = expectRecord(algorithm.prediction, "config_schema.algorithm.prediction");
  if (!Array.isArray(record.sections)) {
    throw new ConfigContractError("config_schema.sections", "array");
  }
  const sections = record.sections.map((sectionValue, sectionIndex): ConfigSectionSchema => {
    const sectionPath = `config_schema.sections[${sectionIndex}]`;
    const section = expectRecord(sectionValue, sectionPath);
    if (!Array.isArray(section.fields)) {
      throw new ConfigContractError(`${sectionPath}.fields`, "array");
    }
    return {
      id: expectString(section.id, `${sectionPath}.id`),
      label: expectString(section.label, `${sectionPath}.label`),
      fields: section.fields.map((fieldValue, fieldIndex): ConfigFieldSchema => {
        const fieldPath = `${sectionPath}.fields[${fieldIndex}]`;
        const field = expectRecord(fieldValue, fieldPath);
        return {
          path: expectString(field.path, `${fieldPath}.path`),
          label: expectString(field.label, `${fieldPath}.label`),
          type: expectLiteral(field.type, `${fieldPath}.type`, ["string", "int", "float", "select", "bool"]),
          options: expectOptionalStringArray(field.options, `${fieldPath}.options`),
          min: expectOptionalNumber(field.min, `${fieldPath}.min`),
          max: expectOptionalNumber(field.max, `${fieldPath}.max`),
          unit: expectOptionalString(field.unit, `${fieldPath}.unit`),
          description: expectOptionalString(field.description, `${fieldPath}.description`),
          apply_mode: expectLiteral(field.apply_mode, `${fieldPath}.apply_mode`, [
            "hot_update", "epoch_reload", "process_restart"
          ]),
          restart_required: expectBoolean(field.restart_required, `${fieldPath}.restart_required`)
        };
      })
    };
  });
  return {
    version: expectUnsignedInteger(record.version, "config_schema.version"),
    algorithm: {
      id: expectString(algorithm.id, "config_schema.algorithm.id"),
      label: expectString(algorithm.label, "config_schema.algorithm.label"),
      response: {
        formula: expectString(response.formula, "config_schema.algorithm.response.formula"),
        radial_multiplier_formula: expectString(
          response.radial_multiplier_formula,
          "config_schema.algorithm.response.radial_multiplier_formula"
        ),
        atan_scale_counts: expectFiniteNumber(
          response.atan_scale_counts,
          "config_schema.algorithm.response.atan_scale_counts"
        )
      },
      prediction: {
        model: expectString(prediction.model, "config_schema.algorithm.prediction.model"),
        aim_history_points: expectUnsignedInteger(
          prediction.aim_history_points,
          "config_schema.algorithm.prediction.aim_history_points"
        ),
        velocity_segments: expectUnsignedInteger(
          prediction.velocity_segments,
          "config_schema.algorithm.prediction.velocity_segments"
        )
      }
    },
    values: decodeRuntimeConfig(record.values),
    sections
  };
}

export function decodeConfigUpdate(value: unknown): ConfigUpdateResponse {
  const record = expectRecord(value, "config_update");
  return {
    config: decodeRuntimeConfig(record.config),
    apply_mode: expectLiteral(record.apply_mode, "config_update.apply_mode", [
      "hot_update", "epoch_reload", "process_restart"
    ]),
    restart_required: expectBoolean(record.restart_required, "config_update.restart_required"),
    applied: expectBoolean(record.applied, "config_update.applied"),
    rolled_back: expectBoolean(record.rolled_back, "config_update.rolled_back"),
    message: expectString(record.message, "config_update.message")
  };
}
