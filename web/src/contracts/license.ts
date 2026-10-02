export type LicenseFeature =
  | "capture"
  | "runtime"
  | "models"
  | "plugins"
  | "tensorrt"
  | "hardware_control"
  | "config_read"
  | "config_write";

export interface LicenseStatus {
  configured: boolean;
  valid: boolean;
  temporary_access_supported: boolean;
  fingerprint: string;
  tier: string;
  features: LicenseFeature[];
  license_id: string;
  credential_format: string;
  token_id: string;
  key_id: string;
  created_at: number | null;
  not_before: number | null;
  activated_at: number | null;
  expires_at: number | null;
  duration_value: number | null;
  duration_unit: string;
  updated_at: number | null;
  message: string;
}

const LICENSE_FEATURES = new Set<LicenseFeature>([
  "capture",
  "runtime",
  "models",
  "plugins",
  "tensorrt",
  "hardware_control",
  "config_read",
  "config_write"
]);

export class LicenseContractError extends Error {
  constructor(path: string, expected: string) {
    super(`授权数据契约错误：${path} 应为 ${expected}`);
    this.name = "LicenseContractError";
  }
}

function expectRecord(value: unknown, path: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new LicenseContractError(path, "object");
  }
  return value as Record<string, unknown>;
}

function expectString(value: unknown, path: string): string {
  if (typeof value !== "string") throw new LicenseContractError(path, "string");
  return value;
}

function expectBoolean(value: unknown, path: string): boolean {
  if (typeof value !== "boolean") throw new LicenseContractError(path, "boolean");
  return value;
}

function expectNullableNumber(value: unknown, path: string): number | null {
  if (value === null) return null;
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new LicenseContractError(path, "finite number | null");
  }
  return value;
}

function expectNullableUnsignedInteger(value: unknown, path: string): number | null {
  if (value === null) return null;
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0) {
    throw new LicenseContractError(path, "unsigned safe integer | null");
  }
  return value;
}

function decodeLicenseStatusAt(value: unknown, path: string): LicenseStatus {
  const record = expectRecord(value, path);
  if (!Array.isArray(record.features)) {
    throw new LicenseContractError(`${path}.features`, "LicenseFeature[]");
  }
  const features = record.features.map((feature, index): LicenseFeature => {
    if (typeof feature !== "string" || !LICENSE_FEATURES.has(feature as LicenseFeature)) {
      throw new LicenseContractError(`${path}.features[${index}]`, [...LICENSE_FEATURES].join(" | "));
    }
    return feature as LicenseFeature;
  });
  return {
    configured: expectBoolean(record.configured, `${path}.configured`),
    valid: expectBoolean(record.valid, `${path}.valid`),
    temporary_access_supported: expectBoolean(
      record.temporary_access_supported,
      `${path}.temporary_access_supported`
    ),
    fingerprint: expectString(record.fingerprint, `${path}.fingerprint`),
    tier: expectString(record.tier, `${path}.tier`),
    features,
    license_id: expectString(record.license_id, `${path}.license_id`),
    credential_format: expectString(record.credential_format, `${path}.credential_format`),
    token_id: expectString(record.token_id, `${path}.token_id`),
    key_id: expectString(record.key_id, `${path}.key_id`),
    created_at: expectNullableNumber(record.created_at, `${path}.created_at`),
    not_before: expectNullableNumber(record.not_before, `${path}.not_before`),
    activated_at: expectNullableNumber(record.activated_at, `${path}.activated_at`),
    expires_at: expectNullableNumber(record.expires_at, `${path}.expires_at`),
    duration_value: expectNullableUnsignedInteger(record.duration_value, `${path}.duration_value`),
    duration_unit: expectString(record.duration_unit, `${path}.duration_unit`),
    updated_at: expectNullableNumber(record.updated_at, `${path}.updated_at`),
    message: expectString(record.message, `${path}.message`)
  };
}

export function decodeLicenseStatus(value: unknown): LicenseStatus {
  return decodeLicenseStatusAt(value, "license");
}
