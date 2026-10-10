import type { LicenseStatus } from "../../contracts/license";

export function formatLicenseCategory(license: LicenseStatus | null): string {
  if (!license?.configured) return license ? "未授权" : "待核实";
  if (license.credential_format === "ephemeral_code" || license.tier === "temporary") return "测试";
  return license.credential_format === "jwt_rs256" || license.credential_format === "legacy_ns1" ? "正式" : "待核实";
}
