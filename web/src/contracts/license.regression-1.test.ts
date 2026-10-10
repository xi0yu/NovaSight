import { describe, expect, it } from "vitest";

import { decodeLicenseStatus } from "./license";

// Regression: ISSUE-001 — legacy plugin feature blocked the license recovery page
// Found by /qa on 2026-08-21
// Report: .gstack/qa-reports/qa-report-novasight-jetson-2026-08-21.md
describe("decodeLicenseStatus legacy feature compatibility", () => {
  it("keeps the legacy plugins marker so an invalid license can still be reactivated", () => {
    const status = decodeLicenseStatus({
      configured: true,
      valid: false,
      temporary_access_supported: true,
      fingerprint: "legacy-license",
      tier: "test_max",
      features: ["capture", "runtime", "models", "plugins", "config_read"],
      license_id: "legacy-license",
      credential_format: "legacy_ns1",
      token_id: "",
      key_id: "",
      created_at: null,
      not_before: null,
      activated_at: null,
      expires_at: null,
      duration_value: null,
      duration_unit: "",
      updated_at: null,
      message: "旧版许可证需要重新激活"
    });

    expect(status.valid).toBe(false);
    expect(status.features).toContain("plugins");
    expect(status.message).toBe("旧版许可证需要重新激活");
  });
});
